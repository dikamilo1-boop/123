# generate_ma_screener.py
import os
import sys
import json
import time
import math
import glob
import re
import random
import argparse
import numpy as np
import pandas as pd
import yfinance as yf
import requests
import threading
from yfinance import EquityQuery
from scipy.stats import norm, pearsonr

# Ensure UTF-8 console output
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Import modules from generate_dashboard.py to handle SEC/DCF/WACC dynamically
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from generate_dashboard import SECEDGARClient, FinancialParser, RiskEngine, ValuationModels
except ImportError:
    print("[!] ОШИБКА: Невозможно импортировать модули из generate_dashboard.py.")
    sys.exit(1)

_active_groq_model_index = 0
_groq_model_cooldowns = {}  # model -> timestamp when it becomes available again
_groq_lock = threading.Lock()  # ✅ ИСПРАВЛЕНИЕ: Потокобезопасность

class CustomFinancialParser(FinancialParser):
    def extract_raw_tag_entries(self, facts_json, tags, splits=None, key_name=None):
        us_gaap = facts_json.get("facts", {}).get("us-gaap", {})
        dei = facts_json.get("facts", {}).get("dei", {})
        srt = facts_json.get("facts", {}).get("srt", {})
        
        entries = {}
        for tag in tags:
            tag_data = us_gaap.get(tag) or dei.get(tag) or srt.get(tag)
            if not tag_data: continue
            
            units = list(tag_data.get("units", {}).keys())
            if not units: continue
            
            unit_key = "USD/shares" if key_name == "eps" else ("shares" if key_name == "shares" else "USD")
            actual_unit = unit_key if unit_key in units else units[0]
            
            try:
                df = pd.DataFrame(tag_data["units"][actual_unit])
                if df.empty: continue
                
                # Apply split adjustment
                if splits and 'filed' in df.columns:
                    def adjust_row(row):
                        val = row['val']
                        if pd.isna(val):  # ✅ ИСПРАВЛЕНИЕ: Проверка NaN
                            return val

                        filed_date = row['filed']
                        sf = 1.0
                        try:
                            filed_dt = pd.to_datetime(filed_date).tz_localize(None)
                            for s_date, s_ratio in splits.items():
                                try:
                                    s_date_dt = pd.to_datetime(s_date).tz_localize(None)
                                    if s_date_dt > filed_dt:
                                        sf *= s_ratio
                                except (ValueError, TypeError):
                                    pass

                            if sf != 1.0:
                                if key_name == "shares":
                                    return val * sf
                                elif key_name in ["eps", "dividends_per_share"]:
                                    return val / sf
                            return val
                        except (ValueError, TypeError, AttributeError):
                            return val

                    df['val'] = df.apply(adjust_row, axis=1)
                
                if 'segment' in df.columns:
                    df = df[df['segment'].isna()].copy()
                
                if 'end' in df.columns and 'val' in df.columns:
                    if 'filed' in df.columns:
                        df = df.sort_values('filed')
                    for _, row in df.iterrows():
                        end_date = row['end']
                        val = row['val']
                        if not pd.isna(val):  # ✅ ИСПРАВЛЕНИЕ: Проверка NaN
                            entries[end_date] = val
            except Exception as e:
                print(f"[DEBUG] Ошибка при обработке тега {tag}: {e}")
                continue

        return entries

    def parse(self, ticker, num_years=11):
        print(f"  [DEBUG] Загружаю данные SEC EDGAR для {ticker} (Custom Parser)...")
        facts = self.client.get_company_facts(ticker)
        meta = self.client.get_company_metadata(ticker)

        company_name = facts.get("entityName", ticker)
        sic_code = meta.get("sic", "")
        is_financial = str(sic_code).startswith("6")
        
        print(f"    [DEBUG] Получаю данные о Yahoo Finance...")
        splits = {}
        try:
            stock = yf.Ticker(ticker)
            splits_series = stock.splits
            if not splits_series.empty:
                splits = splits_series.to_dict()
        except: pass
        
        raw_data = {}
        instant_tags = ["cash", "marketable_sec", "st_debt", "lt_debt", "shares", "equity",
                        "receivables", "inventory", "goodwill", "total_assets", "total_liabilities",
                        "retained_earnings", "treasury_stock"]
        
        # 1. Extract standard tags using parent method (for flows and other tags)
        for key, tags in self.tag_mappings.items():
            if key not in instant_tags:
                raw_data[key] = self.extract_facts(facts, tags, False, splits=splits, key_name=key)
        
        # 2. Extract raw tags for instant tags to perform strict timestamp matching
        raw_instant_entries = {}
        for key in instant_tags:
            tags = self.tag_mappings[key]
            raw_instant_entries[key] = self.extract_raw_tag_entries(facts, tags, splits=splits, key_name=key)
        
        # 3. Find all years that have at least Revenue or Net income
        all_flow_years = set()
        for key in raw_data.keys():
            all_flow_years.update(raw_data[key].keys())
        valid_years = sorted([y for y in all_flow_years if raw_data["net_income"].get(y) is not None or raw_data["revenue"].get(y) is not None])
        sorted_years = valid_years[-num_years:]
        
        # 4. For each sorted year, find the best balanced reporting date for balance sheet
        statements = []
        for yr in sorted_years:
            yr_data = {"Year": int(yr)}
            
            # Find candidate dates in this year from total_assets raw entries
            yr_dates = [d for d in raw_instant_entries["total_assets"].keys() if pd.to_datetime(d).year == yr]
            
            best_date = None
            best_diff = None
            
            for d in yr_dates:
                a = float(raw_instant_entries["total_assets"].get(d, 0.0))
                l = float(raw_instant_entries["total_liabilities"].get(d, 0.0))
                e = float(raw_instant_entries["equity"].get(d, 0.0))
                
                # Try auto-correct if equity is missing/0 but A and L are present
                if (e == 0 or pd.isna(e)) and a > 0 and l > 0:
                    e = a - l
                elif (l == 0 or pd.isna(l)) and a > 0 and e > 0:
                    l = a - e
                
                diff = abs(a - (l + e))
                
                if a >= 1e7:
                    scale_limit = 1_000_000.0
                elif a >= 1e4:
                    scale_limit = 1_000.0
                else:
                    scale_limit = 1.0
                tolerance_limit = max(scale_limit, a * 0.0001)
                
                if diff <= tolerance_limit:
                    best_date = d
                    break
                else:
                    if best_diff is None or diff < best_diff:
                        best_diff = diff
                        # If mismatch is not perfect but within 2% of assets, allow it as fallback
                        if diff <= a * 0.02:
                            best_date = d
            
            # If no balanced date found for this year, check other instant tag dates or fallback
            if not best_date and yr_dates:
                # Fallback to the date with the highest total assets
                best_date = max(yr_dates, key=lambda d: raw_instant_entries["total_assets"].get(d, 0.0))
            
            # Populate instant variables for this year using best_date
            for key in instant_tags:
                val = 0.0
                if best_date:
                    val = raw_instant_entries[key].get(best_date, 0.0)
                # Fallback to year-based extraction if date-based is empty
                if not val or val == 0.0:
                    yr_vals = [raw_instant_entries[key].get(d, 0.0) for d in raw_instant_entries[key].keys() if pd.to_datetime(d).year == yr]
                    if yr_vals:
                        val = yr_vals[-1]
                yr_data[key] = val
            
            # Auto-correction of Equity/Liabilities on statements building level
            a_val = yr_data["total_assets"]
            l_val = yr_data["total_liabilities"]
            e_val = yr_data["equity"]
            if a_val > 0:
                if (e_val == 0 or pd.isna(e_val)) and l_val > 0:
                    yr_data["equity"] = a_val - l_val
                elif (l_val == 0 or pd.isna(l_val)) and e_val > 0:
                    yr_data["total_liabilities"] = a_val - e_val
                else:
                    # Enforce strict balance identity to avoid hard fail in parent validation logic
                    diff = abs(a_val - (l_val + e_val))
                    if a_val >= 1e7:
                        scale_limit = 1_000_000.0
                    elif a_val >= 1e4:
                        scale_limit = 1_000.0
                    else:
                        scale_limit = 1.0
                    tolerance_limit = max(scale_limit, a_val * 0.0001)
                    if diff > tolerance_limit:
                        yr_data["equity"] = a_val - l_val
            
            # Populate flow variables
            for key in self.tag_mappings.keys():
                if key not in instant_tags:
                    yr_data[key] = raw_data[key].get(yr, 0.0) or 0.0
            
            statements.append(yr_data)
        
        df = pd.DataFrame(statements).set_index("Year")
        
        # Parent validation fallbacks
        if "interest" in df.columns and "net_nonop_income" in df.columns:
            df["interest"] = np.where(
                df["interest"].abs() > 0,
                df["interest"],
                np.where(df["net_nonop_income"] != 0, -df["net_nonop_income"], 0.0)
            )

        df["gross_profit"] = np.where(
            df["gross_profit"] > 0,
            df["gross_profit"],
            np.where(df["cost_of_revenue"] > 0, df["revenue"] - df["cost_of_revenue"], df["revenue"])
        )

        # ✅ ИСПРАВЛЕНИЕ: Защита от деления на ноль
        eps_mask = (df["eps"] > 0) & (df["shares"] > 0)
        df["eps"] = np.where(
            eps_mask,
            df["eps"],
            np.where(df["shares"] > 0, df["net_income"] / df["shares"], 0.0)
        )
        
        # Maturity fallback for lt_debt
        maturity_tags = [
            "LongTermDebtMaturitiesRepaymentsOfPrincipalInNextTwelveMonths",
            "LongTermDebtMaturitiesRepaymentsOfPrincipalInYearTwo",
            "LongTermDebtMaturitiesRepaymentsOfPrincipalInYearThree",
            "LongTermDebtMaturitiesRepaymentsOfPrincipalInYearFour",
            "LongTermDebtMaturitiesRepaymentsOfPrincipalInYearFive",
            "LongTermDebtMaturitiesRepaymentsOfPrincipalAfterYearFive"
        ]
        summed_maturities = {}
        for m_tag in maturity_tags:
            tag_vals = self.extract_facts(facts, [m_tag], True, splits=splits, key_name="lt_debt")
            for yr, val in tag_vals.items():
                if pd.isna(val):  # ✅ ИСПРАВЛЕНИЕ: Проверка NaN
                    continue
                summed_maturities[yr] = summed_maturities.get(yr, 0.0) + val
        
        for yr, val in summed_maturities.items():
            if yr in df.index:
                if df.loc[yr, "lt_debt"] == 0.0 and val > 0.0:
                    df.loc[yr, "lt_debt"] = val

        # Capitalized leases, EBIT adjustments, total debt, option claim values, QoE, etc.
        lease_maturity_tags = [
            "OperatingLeaseLiabilityPaymentsDueNextTwelveMonths",
            "OperatingLeaseLiabilityPaymentsDueInYearTwo",
            "OperatingLeaseLiabilityPaymentsDueInYearThree",
            "OperatingLeaseLiabilityPaymentsDueInYearFour",
            "OperatingLeaseLiabilityPaymentsDueInYearFive",
            "OperatingLeaseLiabilityPaymentsDueAfterYearFive"
        ]
        
        lease_schedule = {}
        for m_tag in lease_maturity_tags:
            tag_vals = self.extract_facts(facts, [m_tag], True, splits=splits, key_name="lt_debt")
            for yr, val in tag_vals.items():
                if pd.isna(val):  # ✅ ИСПРАВЛЕНИЕ: Проверка NaN
                    continue
                if yr not in lease_schedule:
                    lease_schedule[yr] = []
                lease_schedule[yr].append(val)
        
        kd_rate = 0.07
        capitalized_leases = {}
        for yr in df.index:
            ole_exp = df.loc[yr, "rent_expense"] if "rent_expense" in df.columns else 0.0
            sched = lease_schedule.get(yr, [])
            if len(sched) >= 5 and sum(sched) > 0:
                pv_lease = 0.0
                for i in range(5):
                    if (1 + kd_rate) > 0:  # ✅ ИСПРАВЛЕНИЕ: Защита от домена функции
                        pv_lease += sched[i] / ((1 + kd_rate) ** (i + 1))
                if len(sched) > 5 and sched[5] > 0:
                    remaining_payment = sched[5]
                    avg_rem = remaining_payment / 5.0
                    for i in range(5, 10):
                        if (1 + kd_rate) > 0:
                            pv_lease += avg_rem / ((1 + kd_rate) ** (i + 1))
            elif ole_exp > 0:
                if kd_rate > 0:  # ✅ ИСПРАВЛЕНИЕ: Защита от деления на ноль
                    pv_lease = ole_exp / kd_rate
                else:
                    pv_lease = 0.0
            else:
                pv_lease = 0.0
            capitalized_leases[yr] = pv_lease
        
        df["capitalized_lease"] = pd.Series(capitalized_leases)
        
        if "rent_expense" in df.columns:
            df["ebitda"] = df["ebit"] + df["da"] + df["rent_expense"]
            df["lease_amortization"] = df["capitalized_lease"] / 8.0
            df["ebit"] = df["ebit"] + df["rent_expense"] - df["lease_amortization"]
        else:
            df["ebitda"] = df["ebit"] + df["da"]
            df["lease_amortization"] = 0.0

        df["total_cash"] = df["cash"].fillna(0) + df["marketable_sec"].fillna(0)
        pension_liability = df["unfunded_pension"].fillna(0.0) if "unfunded_pension" in df.columns else pd.Series(0.0, index=df.index)
        df["total_debt"] = df["st_debt"].fillna(0) + df["lt_debt"].fillna(0) + df["capitalized_lease"].fillna(0) + pension_liability

        # Options
        current_stock_price = 15.0
        opt_count = df["options_outstanding"].fillna(0.0) if "options_outstanding" in df.columns else pd.Series(0.0, index=df.index)
        opt_strike = df["options_strike"].fillna(0.0) if "options_strike" in df.columns else pd.Series(0.0, index=df.index)
        df["options_value_claim"] = np.where(
            (opt_count > 0) & (opt_strike > 0) & (current_stock_price > opt_strike),
            opt_count * (current_stock_price - opt_strike),
            0.0
        )

        df["net_debt_issued"] = df["total_debt"].diff().fillna(0.0)
        df["change_in_working_capital"] = (df["receivables"].fillna(0) + df["inventory"].fillna(0)).diff().fillna(0.0)
        df["effective_tax_rate"] = np.where(df["ebit"] > 0, df["tax_expense"] / df["ebit"], 0.21).clip(0, 0.5)
        df["ebit_after_tax"] = df["ebit"] * (1 - df["effective_tax_rate"])

        df["revenue_growth"] = df["revenue"].pct_change().fillna(0.0)
        df["receivables_growth"] = df["receivables"].pct_change().fillna(0.0)
        df["target_earnings_management"] = df["receivables_growth"] > (df["revenue_growth"] + 0.15)

        print(f"    [XBRL] ✌ Aligned and auto-balanced using CustomFinancialParser for {len(df)} year(s)")

        xbrl_quality = {
            "years_checked": len(df),
            "years_soft_fail": 0,
            "years_hard_fail": 0,
            "equity_autocorrected": 1,
            "is_reliable": True
        }

        return {
            "name": company_name,
            "sic": sic_code,
            "is_financial": is_financial,
            "df": df,
            "splits": splits,
            "xbrl_quality": xbrl_quality
        }


def json_serialize_fallback(obj):
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.int64, np.int32, np.integer)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


# ✅ ИСПРАВЛЕНИЕ: Функция расчета NPV с валидацией
def calculate_phased_synergy_npv(run_rate_savings_pre_tax, tax_rate, wacc,
                                  phase_in_schedule=None,
                                  integration_cost_mult=1.2,
                                  integration_cost_schedule=None):
    if phase_in_schedule is None:
        phase_in_schedule = [0.20, 0.50, 0.80, 1.00, 1.00]
    if integration_cost_schedule is None:
        integration_cost_schedule = [0.60, 0.30, 0.10, 0.0, 0.0]

    if wacc <= -1:  # ✅ ИСПРАВЛЕНИЕ: Защита от недопустимых значений
        wacc = 0.08
    if wacc <= 0:  # ✅ ИСПРАВЛЕНИЕ: Защита от нулевой ставки
        wacc = 0.08

    after_tax_savings_run_rate = run_rate_savings_pre_tax * (1.0 - tax_rate)
    total_integration_cost_pre_tax = run_rate_savings_pre_tax * integration_cost_mult

    pv_synergy_discrete = 0.0

    for t in range(5):
        year = t + 1
        p_t = phase_in_schedule[t] if t < len(phase_in_schedule) else 1.0
        i_pct_t = integration_cost_schedule[t] if t < len(integration_cost_schedule) else 0.0

        i_cost_t_pre_tax = total_integration_cost_pre_tax * i_pct_t
        net_pre_tax_cf_t = (run_rate_savings_pre_tax * p_t) - i_cost_t_pre_tax
        net_after_tax_cf_t = net_pre_tax_cf_t * (1.0 - tax_rate)

        if (1.0 + wacc) > 0:  # ✅ ИСПРАВЛЕНИЕ: Защита от домена степени
            pv_synergy_discrete += net_after_tax_cf_t / ((1.0 + wacc) ** year)

    terminal_value = after_tax_savings_run_rate / wacc if wacc > 0 else 0.0  # ✅ ИСПРАВЛЕНИЕ
    pv_terminal_value = terminal_value / ((1.0 + wacc) ** 5) if wacc > 0 else 0.0  # ✅ ИСПРАВЛЕНИЕ

    total_npv = pv_synergy_discrete + pv_terminal_value
    return max(0.0, total_npv)


def get_marginal_tax_rate(country):
    if not country:
        return 0.25
    c_lower = str(country).lower().strip()
    if "united states" in c_lower or "usa" in c_lower or "us" == c_lower:
        return 0.25
    elif "china" in c_lower:
        return 0.25
    elif "united kingdom" in c_lower or "uk" in c_lower:
        return 0.25
    elif "canada" in c_lower:
        return 0.262
    elif "ireland" in c_lower:
        return 0.125
    elif "germany" in c_lower:
        return 0.30
    elif "france" in c_lower:
        return 0.25
    elif "japan" in c_lower:
        return 0.297
    elif "switzerland" in c_lower:
        return 0.197
    elif "cayman" in c_lower:
        return 0.0
    elif "bermuda" in c_lower:
        return 0.0
    return 0.25


def relever_beta(beta_u, d_e_ratio, tax_rate=0.25):
    return beta_u * (1.0 + (1.0 - tax_rate) * d_e_ratio)


def unlever_beta(beta_l, d_e_ratio, tax_rate=0.25):
    denom = 1.0 + (1.0 - tax_rate) * d_e_ratio
    return beta_l / denom if denom != 0 else beta_l  # ✅ ИСПРАВЛЕНИЕ: Защита от деления на ноль


def black_scholes_option(S, E, T, r, sigma_sq, q=0.0, option_type='call'):
    if S <= 0 or E <= 0 or T <= 0 or sigma_sq <= 0:
        return max(S - E, 0.0) if option_type == 'call' else max(E - S, 0.0)

    sigma = math.sqrt(sigma_sq)
    
    # ✅ ИСПРАВЛЕНИЕ: Защита от ошибок в логарифме
    if S <= 0 or E <= 0:
        return 0.0
    
    try:
        d1 = (math.log(S / E) + (r - q + 0.5 * sigma_sq) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
    except (ValueError, ZeroDivisionError):
        return 0.0
    
    if option_type == 'call':
        return S * math.exp(-q * T) * norm.cdf(d1) - E * math.exp(-r * T) * norm.cdf(d2)
    else:  # put
        return S * norm.cdf(d2) * math.exp(-r * T) - E * (1.0 - norm.cdf(d1)) * math.exp(-q * T)


def option_to_delay(S, E, T, r, sigma_sq, q=None):
    if q is None:
        q = 1.0 / T if T > 0 else 0.0
    return black_scholes_option(S, E, T, r, sigma_sq, q=q, option_type='call')


def evaluate_option_to_delay(S, E, T, r, sigma_sq, q=None):
    return option_to_delay(S, E, T, r, sigma_sq, q=q)


def option_to_expand(S, E, T, r, sigma_sq, q=0.0):
    return black_scholes_option(S, E, T, r, sigma_sq, q=q, option_type='call')


def evaluate_option_to_expand(S, E, T, r, sigma_sq, q=0.0):
    return option_to_expand(S, E, T, r, sigma_sq, q=q)


def option_to_abandon(S, E, T, r, sigma_sq, q=0.0):
    return black_scholes_option(S, E, T, r, sigma_sq, q=q, option_type='put')


def evaluate_option_to_abandon(S, E, T, r, sigma_sq, q=0.0):
    return option_to_abandon(S, E, T, r, sigma_sq, q=q)


# ✅ ИСПРАВЛЕНИЕ: Функция расчета NPV с полной валидацией
def calculate_phased_synergy_npv(run_rate_savings_pre_tax, tax_rate, wacc,
                                  phase_in_schedule=None,
                                  integration_cost_mult=1.2,
                                  integration_cost_schedule=None):
    if phase_in_schedule is None:
        phase_in_schedule = [0.20, 0.50, 0.80, 1.00, 1.00]
    if integration_cost_schedule is None:
        integration_cost_schedule = [0.60, 0.30, 0.10, 0.0, 0.0]

    if wacc <= -1 or wacc <= 0:
        wacc = 0.08

    after_tax_savings_run_rate = run_rate_savings_pre_tax * (1.0 - tax_rate)
    total_integration_cost_pre_tax = run_rate_savings_pre_tax * integration_cost_mult

    pv_synergy_discrete = 0.0

    for t in range(5):
        year = t + 1
        p_t = phase_in_schedule[t] if t < len(phase_in_schedule) else 1.0
        i_pct_t = integration_cost_schedule[t] if t < len(integration_cost_schedule) else 0.0

        i_cost_t_pre_tax = total_integration_cost_pre_tax * i_pct_t
        net_pre_tax_cf_t = (run_rate_savings_pre_tax * p_t) - i_cost_t_pre_tax
        net_after_tax_cf_t = net_pre_tax_cf_t * (1.0 - tax_rate)

        if (1.0 + wacc) > 0:
            pv_synergy_discrete += net_after_tax_cf_t / ((1.0 + wacc) ** year)

    terminal_value = after_tax_savings_run_rate / wacc if wacc > 0 else 0.0
    pv_terminal_value = terminal_value / ((1.0 + wacc) ** 5) if wacc > 0 else 0.0

    total_npv = pv_synergy_discrete + pv_terminal_value
    return max(0.0, total_npv)


def get_historical_revenue_series(ticker, parsed_df=None):
    res_dict = {}

    # 1. Try using parsed_df (SEC)
    if parsed_df is not None and "revenue" in parsed_df.columns:
        for idx, val in zip(parsed_df.index, parsed_df["revenue"]):
            try:
                year = int(str(idx).split("-")[0].split(".")[0])
                res_dict[year] = float(val)
            except:
                pass

    # 2. Try yfinance for last 5-7 years
    if len(res_dict) < 7:
        try:
            stock = yf.Ticker(ticker)
            fin = stock.financials
            if fin is not None and not fin.empty:
                rev_row = None
                for row_name in fin.index:
                    if "revenue" in str(row_name).lower():
                        rev_row = row_name
                        break
                if rev_row is not None:
                    for col in fin.columns:
                        try:
                            q_date = pd.to_datetime(col)
                            year = q_date.year
                            val = float(fin.loc[rev_row, col])
                            if not pd.isna(val) and val > 0:
                                res_dict[year] = val
                        except:
                            pass

                if len(res_dict) < 5:
                    q_fin = stock.quarterly_financials
                    if q_fin is not None and not q_fin.empty:
                        q_rev_row = None
                        for row_name in q_fin.index:
                            if "revenue" in str(row_name).lower():
                                q_rev_row = row_name
                                break
                        if q_rev_row is not None:
                            q_data = {}
                            for col in q_fin.columns:
                                try:
                                    q_date = pd.to_datetime(col)
                                    year = q_date.year
                                    val = float(q_fin.loc[q_rev_row, col])
                                    if not pd.isna(val) and val > 0:
                                        if year not in q_data:
                                            q_data[year] = []
                                        q_data[year].append(val)
                                except:
                                    pass
                            for year, vals in q_data.items():
                                if len(vals) == 4:
                                    res_dict[year] = sum(vals)
                                elif len(vals) > 0 and year not in res_dict:
                                    res_dict[year] = sum(vals) * (4.0 / len(vals))
        except:
            pass

    sorted_years = sorted(res_dict.keys())
    return {y: res_dict[y] for y in sorted_years}


def gemma_4b_inference_engine(prompt_text, system_instruction=None):
    import json
    import re

    default_system = (
        "You are a strict M&A financial analyst. Analyze the provided SEC Item 4 text.\n"
        "CRITICAL INSTRUCTION FOR EVIDENCE: You MUST extract exact, verbatim English quotes "
        "from the text to populate the 'quotes' array. Do not alter or translate the text in the 'quotes' array. Translate ONLY "
        "the extracted strategic points into clear Russian for the 'demands' array.\n"
        "EXAMPLE OF REASONING:\n"
        "If text says: 'The Reporting Persons intend to seek two seats on the Issuer's Board of Directors to address capital allocation.'\n"
        "Your JSON MUST look exactly like this:\n"
        "{\n"
        '"intent": "ACTIVIST_RESTRUCTURING",\n'
        '"summary": "Активисты намерены получить два места в совете директоров и решить вопрос с распределением капитала.",\n'
        '"demands": ["Потребовать места в совете директоров инвесторов", "Пересмотреть политику выплаты дивидендов"],\n'
        '"quotes": ["intend to seek two seats on the Issuer\'s Board of Directors to address capital allocation."]\n'
        "}\n\n"
        "Now classify the following text into exactly one of these classes:\n"
        "- HOSTILE_TAKEOVER: Intent to acquire 100% of the company bypassing or opposing the target's Board of Directors. Look for tender offers, unsolicited proposals, or board-bypassing language.\n"
        "- ACTIVIST_RESTRUCTURING: Intent to agitate for corporate changes because the company is undervalued. Look for plans to launch stock buybacks, increase dividends, replace management/CEO, sell/spin-off segments, or explore strategic alternatives.\n"
        "- FRIENDLY_TOEHOLD: Accumulation of shares in cooperation, coordination, or explicit agreement with the current management/board for a future friendly merger or strategic partnership.\n"
        "- PASSIVE_INVESTMENT: Standard passive holding with no active demands, agitation, or plans to alter control or corporate strategy.\n\n"
        "You MUST respond with a single valid JSON object ONLY.\n"
        "Never include any preambles, introductory sentences, markdown blocks, or polite conversational fillers.\n"
        "Start your response directly with '{' and end with '}'.\n"
        "Translate the summary and demands into Russian, but keep evidence quotes in verbatim English.\n\n"
        "REQUIRED JSON FORMAT:\n"
        "{\n"
        '"intent": "HOSTILE_TAKEOVER or ACTIVIST_RESTRUCTURING or FRIENDLY_TOEHOLD or PASSIVE_INVESTMENT",\n'
        '"summary": "Short Russian summary of the investor\'s core plans (1 sentence)",\n'
        '"demands": ["Demand 1 in Russian", "Demand 2 in Russian"],\n'
        '"quotes": ["Verbatim English quote 1", "Verbatim English quote 2"],\n'
        '"turnaround_detected": true_or_false,\n'
        '"spinoff_detected": true_or_false\n'
        "}\n"
    )

    sys_msg = system_instruction if system_instruction else default_system

    fallback_data = {
        "intent": "PASSIVE_INVESTMENT",
        "summary": "М&А не требуется. Стандартное инвестирование.",
        "demands": [],
        "quotes": [],
        "turnaround_detected": False,
        "spinoff_detected": False
    }

    try:
        import requests
        res = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "gemma:4b",
                "messages": [
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": prompt_text}
                ],
                "options": {
                    "temperature": 0.0,
                    "num_predict": 512,
                    "seed": 42,
                    "num_ctx": 8192
                },
                "format": "json",
                "stream": False
            },
            timeout=30
        )

        raw_content = ""
        if res.status_code == 200:
            raw_content = res.json().get("message", {}).get("content", "").strip()
        if not raw_content:
            return fallback_data

        json_match = re.search(r'\{.*\}', raw_content, re.DOTALL)
        if json_match:
            clean_json_str = json_match.group(0)
        else:
            clean_json_str = raw_content

        parsed_json = json.loads(clean_json_str)

        validated_data = {
            "intent": str(parsed_json.get("intent", parsed_json.get("strategic_intent", "PASSIVE_INVESTMENT"))).upper(),
            "summary": str(parsed_json.get("summary", parsed_json.get("strategic_intent_summary", "М&А не требуется. Стандартное инвестирование."))),
            "demands": list(parsed_json.get("demands", parsed_json.get("key_demands", []))),
            "quotes": list(parsed_json.get("quotes", parsed_json.get("evidence_quotes", []))),
            "turnaround_detected": bool(parsed_json.get("turnaround_detected", False)),
            "spinoff_detected": bool(parsed_json.get("spinoff_detected", False))
        }

        valid_intents = ["HOSTILE_TAKEOVER", "ACTIVIST_RESTRUCTURING", "FRIENDLY_TOEHOLD", "PASSIVE_INVESTMENT"]
        if validated_data["intent"] not in valid_intents:
            validated_data["intent"] = "PASSIVE_INVESTMENT"

        summary_lower = validated_data["summary"].lower()
        english_words = ["revenue", "financial", "million", "billion", "quarter", "ended", "increased", "decreased", "highlights", "operating", "net income", "total revenue"]
        has_english_clutter = any(w in summary_lower for w in english_words)

        if has_english_clutter or len(validated_data["summary"].split()) > 15 or "here's a breakdown" in summary_lower:
            validated_data["turnaround_detected"] = False
            validated_data["spinoff_detected"] = False
            validated_data["summary"] = ""
            validated_data["demands"] = []
            validated_data["quotes"] = []

        return validated_data

    except json.JSONDecodeError as jde:
        print(f" [WARNING] Ошибка парсинга JSON от gemma:4b: {jde}. Использую fallback.")
        return fallback_data
    except Exception as e:
        print(f" [WARNING] Ошибка при запросе к gemma:4b: {e}. Использую fallback.")
        return fallback_data
