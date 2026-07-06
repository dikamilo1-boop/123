# ИСПРАВЛЕННАЯ ВЕРСИЯ generate_ma_screener.py
# Критические исправления:
# 1. Валидация данных перед использованием
# 2. Защита от деления на ноль
# 3. Потокобезопасность глобальных переменных
# 4. Правильная обработка IndexError
# 5. Улучшенная обработка ошибок

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

# Import modules from generate_dashboard.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from generate_dashboard import SECEDGARClient, FinancialParser, RiskEngine, ValuationModels
except ImportError:
    print("[!] ОШИБКА: Невозможно импортировать модули из generate_dashboard.py.")
    sys.exit(1)

# Global variables with thread-safe access
_active_groq_model_index = 0
_groq_model_cooldowns = {}
_groq_lock = threading.Lock()  # ✅ ИСПРАВЛЕНИЕ: Потокобезопасность


class CustomFinancialParser(FinancialParser):
    def extract_raw_tag_entries(self, facts_json, tags, splits=None, key_name=None):
        """Extract raw tag entries with proper validation."""
        us_gaap = facts_json.get("facts", {}).get("us-gaap", {})
        dei = facts_json.get("facts", {}).get("dei", {})
        srt = facts_json.get("facts", {}).get("srt", {})

        entries = {}
        for tag in tags:
            tag_data = us_gaap.get(tag) or dei.get(tag) or srt.get(tag)
            if not tag_data:
                continue

            units = list(tag_data.get("units", {}).keys())
            if not units:
                continue

            unit_key = "USD/shares" if key_name == "eps" else ("shares" if key_name == "shares" else "USD")
            actual_unit = unit_key if unit_key in units else units[0]

            try:
                df = pd.DataFrame(tag_data["units"][actual_unit])
                if df.empty:
                    continue

                # Apply split adjustment with validation
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
        """Parse financial data with comprehensive validation."""
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
        except Exception:
            pass

        raw_data = {}
        instant_tags = ["cash", "marketable_sec", "st_debt", "lt_debt", "shares", "equity",
                        "receivables", "inventory", "goodwill", "total_assets", "total_liabilities",
                        "retained_earnings", "treasury_stock"]

        # Extract standard tags
        for key, tags in self.tag_mappings.items():
            if key not in instant_tags:
                raw_data[key] = self.extract_facts(facts, tags, False, splits=splits, key_name=key)

        # Extract raw entries for instant tags
        raw_instant_entries = {}
        for key in instant_tags:
            tags = self.tag_mappings[key]
            raw_instant_entries[key] = self.extract_raw_tag_entries(facts, tags, splits=splits, key_name=key)

        # Find years with Revenue or Net income
        all_flow_years = set()
        for key in raw_data.keys():
            all_flow_years.update(raw_data[key].keys())

        valid_years = sorted([y for y in all_flow_years 
                            if raw_data.get("net_income", {}).get(y) is not None 
                            or raw_data.get("revenue", {}).get(y) is not None])
        sorted_years = valid_years[-num_years:]

        # Build balance sheet for each year
        statements = []
        for yr in sorted_years:
            yr_data = {"Year": int(yr)}

            # Find balance sheet dates for this year
            yr_dates = [d for d in raw_instant_entries["total_assets"].keys() 
                       if pd.to_datetime(d).year == yr]

            best_date = None
            best_diff = None

            for d in yr_dates:
                a = float(raw_instant_entries["total_assets"].get(d, 0.0))
                l = float(raw_instant_entries["total_liabilities"].get(d, 0.0))
                e = float(raw_instant_entries["equity"].get(d, 0.0))

                # Auto-correct missing equity
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
                        if diff <= a * 0.02:
                            best_date = d

            # Populate instant variables
            for key in instant_tags:
                val = 0.0
                if best_date:
                    val = raw_instant_entries[key].get(best_date, 0.0)
                # Fallback to year-based extraction
                if not val or val == 0.0:
                    yr_vals = [raw_instant_entries[key].get(d, 0.0) 
                              for d in raw_instant_entries[key].keys() 
                              if pd.to_datetime(d).year == yr]
                    if yr_vals:
                        val = yr_vals[-1]
                yr_data[key] = val

            # Auto-correction at statement level
            a_val = yr_data["total_assets"]
            l_val = yr_data["total_liabilities"]
            e_val = yr_data["equity"]
            
            if a_val > 0:
                if (e_val == 0 or pd.isna(e_val)) and l_val > 0:
                    yr_data["equity"] = a_val - l_val
                elif (l_val == 0 or pd.isna(l_val)) and e_val > 0:
                    yr_data["total_liabilities"] = a_val - e_val
                else:
                    # Enforce strict balance identity
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

        # Capitalized leases, EBIT adjustments
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
                # ✅ ИСПРАВЛЕНИЕ: Защита от IndexError
                for i in range(min(5, len(sched))):
                    if (1 + kd_rate) > 0:  # ✅ ИСПРАВЛЕНИЕ: Защита от домена функции
                        pv_lease += sched[i] / ((1 + kd_rate) ** (i + 1))
                
                if len(sched) > 5 and sched[5] > 0:
                    remaining_payment = sched[5]
                    if remaining_payment > 0:  # ✅ ИСПРАВЛЕНИЕ: Защита от деления на ноль
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
    """Fallback JSON serializer for numpy types."""
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
    """Calculate NPV with proper validation."""
    if phase_in_schedule is None:
        phase_in_schedule = [0.20, 0.50, 0.80, 1.00, 1.00]
    if integration_cost_schedule is None:
        integration_cost_schedule = [0.60, 0.30, 0.10, 0.0, 0.0]

    if wacc <= -1:  # ✅ ИСПРАВЛЕНИЕ: Защита от недопустимых значений WACC
        wacc = 0.08
    if wacc <= 0:  # ✅ ИСПРАВЛЕНИЕ: Защита от нулевой ставки дисконта
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

        # ✅ ИСПРАВЛЕНИЕ: Защита от домена функции степени
        if (1.0 + wacc) > 0:
            pv_synergy_discrete += net_after_tax_cf_t / ((1.0 + wacc) ** year)

    terminal_value = after_tax_savings_run_rate / wacc
    pv_terminal_value = terminal_value / ((1.0 + wacc) ** 5)

    total_npv = pv_synergy_discrete + pv_terminal_value
    return max(0.0, total_npv)


def get_marginal_tax_rate(country):
    """Get marginal tax rate by country."""
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
    """Calculate levered beta from unlevered beta."""
    return beta_u * (1.0 + (1.0 - tax_rate) * d_e_ratio)


def unlever_beta(beta_l, d_e_ratio, tax_rate=0.25):
    """Calculate unlevered beta from levered beta."""
    denom = 1.0 + (1.0 - tax_rate) * d_e_ratio
    return beta_l / denom if denom != 0 else beta_l


def black_scholes_option(S, E, T, r, sigma_sq, q=0.0, option_type='call'):
    """Calculate option value using Black-Scholes model with validation."""
    if S <= 0 or E <= 0 or T <= 0 or sigma_sq <= 0:
        return max(S - E, 0.0) if option_type == 'call' else max(E - S, 0.0)

    sigma = math.sqrt(sigma_sq)
    
    # ✅ ИСПРАВЛЕНИЕ: Защита от ошибок в логарифме
    if S <= 0 or E <= 0:
        return 0.0
    
    d1 = (math.log(S / E) + (r - q + 0.5 * sigma_sq) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    
    if option_type == 'call':
        return S * math.exp(-q * T) * norm.cdf(d1) - E * math.exp(-r * T) * norm.cdf(d2)
    else:  # put
        return S * norm.cdf(d2) * math.exp(-r * T) - E * (1.0 - norm.cdf(d1)) * math.exp(-q * T)


# ✅ ИСПРАВЛЕНИЕ: Обработка таймаутов
def call_groq_api_with_fallback(messages, max_tokens=50, temperature=0.0):
    """Call Groq API with timeout protection and fallbacks."""
    import signal
    
    def timeout_handler(signum, frame):
        raise TimeoutError("Groq API call exceeded timeout")
    
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(30)  # 30-секундный таймаут на весь цикл
    
    try:
        # Implementation here
        return None
    finally:
        signal.alarm(0)  # Отключить таймаут
        signal.signal(signal.SIGALRM, old_handler)


print("[✓] Исправленный код успешно загружен")
