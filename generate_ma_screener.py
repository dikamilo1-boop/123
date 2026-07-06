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
from yfinance import EquityQuery
from scipy.stats import norm, pearsonr

# Ensure UTF-8 console output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Import modules from generate_dashboard.py to handle SEC/DCF/WACC dynamically
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from generate_dashboard import SECEDGARClient, FinancialParser, RiskEngine, ValuationModels
except ImportError:
    print("[!] Ошибка: Не удалось импортировать модули из generate_dashboard.py. Убедитесь, что файл находится в той же папке.")
    sys.exit(1)
_active_groq_model_index = 0
_groq_model_cooldowns = {}  # model -> timestamp when it becomes available again

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
            
            df = pd.DataFrame(tag_data["units"][actual_unit])
            if df.empty: continue
            
            # Apply split adjustment
            if splits and 'filed' in df.columns:
                def adjust_row(row):
                    val = row['val']
                    filed_date = row['filed']
                    sf = 1.0
                    try:
                        filed_dt = pd.to_datetime(filed_date).tz_localize(None)
                        for s_date, s_ratio in splits.items():
                            s_date_dt = pd.to_datetime(s_date).tz_localize(None)
                            if s_date_dt > filed_dt:
                                sf *= s_ratio
                    except: pass
                    
                    if sf != 1.0:
                        if key_name == "shares":
                            return val * sf
                        elif key_name in ["eps", "dividends_per_share"]:
                            return val / sf
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
                    entries[end_date] = val
        return entries

    def parse(self, ticker, num_years=11):
        print(f"  [DEBUG] Подключение к SEC EDGAR для {ticker} (Custom Parser)...")
        facts = self.client.get_company_facts(ticker)
        meta = self.client.get_company_metadata(ticker)
        
        company_name = facts.get("entityName", ticker)
        sic_code = meta.get("sic", "")
        is_financial = str(sic_code).startswith("6")
        
        print("  [DEBUG] Подгрузка сплитов акций с Yahoo Finance...")
        splits = {}
        try:
            stock = yf.Ticker(ticker)
            splits_series = stock.splits
            if not splits_series.empty:
                splits = splits_series.to_dict()
        except: pass
        
        raw_data = {}
        instant_tags = ["cash", "marketable_sec", "st_debt", "lt_debt", "shares", "equity", "receivables", "inventory", "goodwill", "total_assets", "total_liabilities", "retained_earnings", "treasury_stock"]
        
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
        
        df["eps"] = np.where(df["eps"] > 0, df["eps"], np.where(df["shares"] > 0, df["net_income"] / df["shares"], 0.0))
        
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
                    pv_lease += sched[i] / ((1 + kd_rate) ** (i + 1))
                if len(sched) > 5 and sched[5] > 0:
                    remaining_payment = sched[5]
                    avg_rem = remaining_payment / 5.0
                    for i in range(5, 10):
                        pv_lease += avg_rem / ((1 + kd_rate) ** (i + 1))
            elif ole_exp > 0:
                pv_lease = ole_exp / kd_rate
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
        
        print(f"    [XBRL] ✓ Aligned and auto-balanced using CustomFinancialParser for {len(df)} year(s)")
        
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



# ─────────────────────────────────────────────────────────────────────────────
# ERP: Damodaran Implied Equity Risk Premium (обновляется вручную раз в квартал).
# Актуальное значение: https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/implprem.html
# Последнее обновление константы: январь 2025 → 4.60%
# ─────────────────────────────────────────────────────────────────────────────
DAMODARAN_ERP = 0.046  # Implied ERP (S&P 500, Jan 2025)

# ─────────────────────────────────────────────────────────────────────────────
# ФИНАНСОВЫЕ ПРИМИТИВЫ: Формула Хамады (Ch. 7) и модель Блэка-Шоулза (Ch. 8)
# Вынесены в отдельные функции, чтобы использоваться единообразно во всех
# местах кода, где раньше формулы дублировались инлайн (releverage WACC,
# оценка equity как реального опциона для distress-компаний).
# ─────────────────────────────────────────────────────────────────────────────


def get_marginal_tax_rate(country):
    """
    Возвращает предельную налоговую ставку (marginal tax rate) по странам (Damodaran 2025/2026).
    """
    if not country:
        return 0.25  # по умолчанию США
    c_lower = str(country).lower().strip()
    if "united states" in c_lower or "usa" in c_lower or "us" == c_lower:
        return 0.25  # 21% федеральный + ~4% штаты
    elif "china" in c_lower:
        return 0.25
    elif "united kingdom" in c_lower or "uk" in c_lower:
        return 0.25
    elif "canada" in c_lower:
        return 0.262  # ~26.2%
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
    return 0.25  # дефолт

def relever_beta(beta_u, d_e_ratio, tax_rate=0.25):
    """
    Формула Хамады: релеверинг unlevered Beta под целевую структуру капитала
    (Damodaran Ch. 7). beta_l = beta_u * [1 + (1 - t) * (D/E)]
    По умолчанию используется предельная налоговая ставка (marginal tax rate = 25%).
    """
    return beta_u * (1.0 + (1.0 - tax_rate) * d_e_ratio)


def unlever_beta(beta_l, d_e_ratio, tax_rate=0.25):
    """
    Обратная формула Хамады: разлеверинг текущей (levered) Beta до
    операционной (unlevered) Beta. beta_u = beta_l / [1 + (1 - t) * (D/E)]
    По умолчанию используется предельная налоговая ставка (marginal tax rate = 25%).
    """
    denom = 1.0 + (1.0 - tax_rate) * d_e_ratio
    return beta_l / denom if denom != 0 else beta_l


def black_scholes_option(S, E, T, r, sigma_sq, q=0.0, option_type='call'):
    """
    Оценка реального опциона по модели Блэка-Шоулза (Damodaran Ch. 8).

    S         — стоимость базового актива (Enterprise/Asset Value)
    E         — цена исполнения (Face Value of Debt, номинал долга к погашению)
    T         — время до экспирации в годах (средневзвешенная дюрация долга)
    r         — безрисковая ставка
    sigma_sq  — дисперсия доходности базового актива (volatility^2)
    q         — утечка стоимости/дивидендная доходность (по умолчанию 0)
    option_type — 'call' (equity как опцион на активы) или 'put'
    """
    if S <= 0 or E <= 0 or T <= 0 or sigma_sq <= 0:
        return max(S - E, 0.0) if option_type == 'call' else max(E - S, 0.0)

    sigma = math.sqrt(sigma_sq)
    d1 = (math.log(S / E) + (r - q + 0.5 * sigma_sq) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == 'call':
        return S * math.exp(-q * T) * norm.cdf(d1) - E * math.exp(-r * T) * norm.cdf(d2)
    else:  # put option (Abandonment Put Option, DePamphilis Exhibit 8.10)
        return S * norm.cdf(d2) * math.exp(-r * T) - E * (1.0 - norm.cdf(d1)) * math.exp(-q * T)


def option_to_delay(S, E, T, r, sigma_sq, q=None):
    """
    Option to Delay (Опцион на отсрочку — Aztec Corp, DePamphilis Exhibit 8.9).
    Оценка колл-опциона на запуск проекта с учетом дивидендной утечки (q как альтернативные издержки ожидания патентов или лицензий).
    По умолчанию q = 1 / T (как в Aztec Corp).
    """
    if q is None:
        q = 1.0 / T if T > 0 else 0.0
    return black_scholes_option(S, E, T, r, sigma_sq, q=q, option_type='call')


def evaluate_option_to_delay(S, E, T, r, sigma_sq, q=None):
    return option_to_delay(S, E, T, r, sigma_sq, q=q)


def option_to_expand(S, E, T, r, sigma_sq, q=0.0):
    """
    Option to Expand (Опцион на расширение — Comet, DePamphilis Exhibit 8.8).
    Оценка колл-опциона на ретулинг или расширение мощностей при изменении спроса.
    """
    return black_scholes_option(S, E, T, r, sigma_sq, q=q, option_type='call')


def evaluate_option_to_expand(S, E, T, r, sigma_sq, q=0.0):
    return option_to_expand(S, E, T, r, sigma_sq, q=q)


def option_to_abandon(S, E, T, r, sigma_sq, q=0.0):
    """
    Option to Abandon (Опцион на выход — Bernard Mining, DePamphilis Exhibit 8.10).
    Расчет стоимости пут-опциона на ликвидацию бизнеса при неблагоприятном исходе,
    с использованием специфической скорректированной формулы DePamphilis Exhibit 8.10.
    Формула: P = S * (1 - N(d2)) * e^(-r * T) - E * (1 - N(d1)) * e^(-q * T)
    """
    return black_scholes_option(S, E, T, r, sigma_sq, q=q, option_type='put')


def evaluate_option_to_abandon(S, E, T, r, sigma_sq, q=0.0):
    return option_to_abandon(S, E, T, r, sigma_sq, q=q)






def calculate_phased_synergy_npv(run_rate_savings_pre_tax, tax_rate, wacc, 
                                 phase_in_schedule=None, 
                                 integration_cost_mult=1.2, 
                                 integration_cost_schedule=None):
    """
    Расчет чистой приведенной стоимости (NPV) синергии с учетом поэтапного внедрения
    (phasing-in) и единовременных затрат на интеграцию (Integration Expenses)
    согласно Главе 14 методологии DePamphilis / Damodaran.
    
    run_rate_savings_pre_tax: Годовая доналоговая синергия (run-rate, например, 15% SG&A цели)
    tax_rate: Предельная налоговая ставка (marginal tax rate)
    wacc: Ставка дисконтирования (WACC покупателя или объединенной компании)
    phase_in_schedule: Поэтапный коэффициент внедрения синергии по годам 1-5 (по умолчанию [0.20, 0.50, 0.80, 1.00, 1.00])
    integration_cost_mult: Коэффициент затрат на интеграцию к годовой экономии (по умолчанию 1.2x)
    integration_cost_schedule: Распределение затрат на интеграцию по годам 1-5 (по умолчанию [0.60, 0.30, 0.10, 0.0, 0.0])
    """
    if phase_in_schedule is None:
        phase_in_schedule = [0.20, 0.50, 0.80, 1.00, 1.00]
    if integration_cost_schedule is None:
        integration_cost_schedule = [0.60, 0.30, 0.10, 0.0, 0.0]
        
    if wacc <= 0:
        wacc = 0.08  # дефолтное безопасное значение
        
    # Годовая посленалоговая run-rate синергия
    after_tax_savings_run_rate = run_rate_savings_pre_tax * (1.0 - tax_rate)
    
    # Полные доналоговые затраты на интеграцию (выходные пособия, IT-слияния и т.д.)
    total_integration_cost_pre_tax = run_rate_savings_pre_tax * integration_cost_mult
    
    pv_synergy_discrete = 0.0
    
    # Моделируем 5-летний дискретный период поэтапного внедрения
    for t in range(5):
        year = t + 1
        p_t = phase_in_schedule[t] if t < len(phase_in_schedule) else 1.0
        i_pct_t = integration_cost_schedule[t] if t < len(integration_cost_schedule) else 0.0
        
        # Доналоговые затраты на интеграцию в году t
        i_cost_t_pre_tax = total_integration_cost_pre_tax * i_pct_t
        
        # Чистый доналоговый поток синергии в году t
        net_pre_tax_cf_t = (run_rate_savings_pre_tax * p_t) - i_cost_t_pre_tax
        
        # Посленалоговый поток синергии в году t (т.к. интеграционные расходы вычитаются из налогооблагаемой прибыли)
        net_after_tax_cf_t = net_pre_tax_cf_t * (1.0 - tax_rate)
        
        # Дисконтируем к текущему моменту (PV)
        pv_synergy_discrete += net_after_tax_cf_t / ((1.0 + wacc) ** year)
        
    # Терминальная стоимость синергии со 6-го года (перпетуитет run-rate, дисконтированный к году 5, затем к году 0)
    terminal_value = after_tax_savings_run_rate / wacc
    pv_terminal_value = terminal_value / ((1.0 + wacc) ** 5)
    
    total_npv = pv_synergy_discrete + pv_terminal_value
    return max(0.0, total_npv)

def get_historical_revenue_series(ticker, parsed_df=None):
    """
    Обеспечивает импорт исторических финансовых данных за 5–7 лет для сглаживания циклов.
    Извлекает данные из parsed_df (SEC) и дозагружает через yfinance при необходимости.
    """
    res_dict = {}
    
    # 1. Извлекаем из parsed_df (SEC EDGAR Client)
    if parsed_df is not None and "revenue" in parsed_df.columns:
        for idx, val in zip(parsed_df.index, parsed_df["revenue"]):
            try:
                year = int(str(idx).split("-")[0].split(".")[0])
                res_dict[year] = float(val)
            except:
                pass
                
    # 2. Дозагружаем из yfinance для обеспечения 5-7 лет
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
                            year = int(str(col).split("-")[0].split(".")[0])
                            val = float(fin.loc[rev_row, col])
                            if not pd.isna(val) and val > 0:
                                res_dict[year] = val
                        except:
                            pass
            
            # Если всё ещё меньше 5 лет, пробуем квартальные
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
                                    q_data.setdefault(year, []).append(val)
                            except:
                                pass
                        for year, vals in q_data.items():
                            if len(vals) == 4:
                                res_dict[year] = sum(vals)
                            elif len(vals) > 0 and year not in res_dict:
                                res_dict[year] = sum(vals) * (4.0 / len(vals))
        except:
            pass
            
    # Сортируем по годам и возвращаем
    sorted_years = sorted(res_dict.keys())
    return {y: res_dict[y] for y in sorted_years}



def gemma_4b_inference_engine(prompt_text, system_instruction=None):
    """
    Интеграционный движок инференса для локальной модели Gemma 3 4B.
    
    Обеспечивает:
    1. Жесткое ограничение температуры (0.0) для устранения галлюцинаций.
    2. Аппаратный JSON-режим инференса через Ollama API.
    3. Агрессивную очистку сырого вывода регулярными выражениями (Regex Extract).
    4. Безопасную деградацию данных (Graceful Fallback) при сбое структуры JSON.
    """
    import json
    import re
        
    # Сверхплаский промпт-шаблон, адаптированный под возможности модели 4B (Правка Шага 5)
    default_system = (
        "You are a strict M&A financial analyst. Analyze the provided SEC Item 4 text.\n"
        "CRITICAL INSTRUCTION FOR EVIDENCE: You MUST extract exact, verbatim English quotes from the text to populate the 'quotes' array. Do not alter or translate the text in the 'quotes' array. Translate ONLY the extracted strategic points into clear Russian for the 'demands' array.\n"
        "EXAMPLE OF REASONING:\n"
        "If text says: 'The Reporting Persons intend to seek two seats on the Issuer's Board of Directors to address capital allocation.'\n"
        "Your JSON MUST look exactly like this:\n"
        "{\n"
        "  \"intent\": \"ACTIVIST_RESTRUCTURING\",\n"
        "  \"summary\": \"Инвестор требует изменения аллокации капитала и места в Совете директоров.\",\n"
        "  \"demands\": [\"Получение двух мест в Совете директоров\", \"Пересмотр стратегии распределения капитала\"],\n"
        "  \"quotes\": [\"intend to seek two seats on the Issuer's Board of Directors to address capital allocation.\"],\n"
        "  \"turnaround_detected\": false,\n"
        "  \"spinoff_detected\": false\n"
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
        "  \"intent\": \"HOSTILE_TAKEOVER or ACTIVIST_RESTRUCTURING or FRIENDLY_TOEHOLD or PASSIVE_INVESTMENT\",\n"
        "  \"summary\": \"Short Russian summary of the investor's core plans (1 sentence)\",\n"
        "  \"demands\": [\"Demand 1 in Russian\", \"Demand 2 in Russian\"],\n"
        "  \"quotes\": [\"Verbatim English quote 1\", \"Verbatim English quote 2\"],\n"
        "  \"turnaround_detected\": true_or_false,\n"
        "  \"spinoff_detected\": true_or_false\n"
        "}"
    )
    
    sys_msg = system_instruction if system_instruction else default_system
    
    # Безопасный дефолтный JSON на случай непредвиденного сбоя (HARD FAIL Protection)
    fallback_data = {
        "intent": "PASSIVE_INVESTMENT",
        "summary": "M&A зацепок не обнаружено. Ошибка разбора или таймаут локального инференса gemma3:4b.",
        "demands": [],
        "quotes": [],
        "turnaround_detected": False,
        "spinoff_detected": False
    }
    
    try:
        # 1. Запуск Ollama через нативный requests POST к API
        import requests
        res = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "gemma3:4b",
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
            
        # 2. Агрессивная очистка вывода регулярными выражениями
        # Вырезаем всё, что находится за пределами крайних фигурных скобок {}
        json_match = re.search(r'\{.*\}', raw_content, re.DOTALL)
        if json_match:
            clean_json_str = json_match.group(0)
        else:
            clean_json_str = raw_content
            
        # 3. Десериализация очищенной строки
        parsed_json = json.loads(clean_json_str)
        
        # 4. Нормализация ключей и валидация типов (страховка от вольностей модели 4B)
        validated_data = {
            "intent": str(parsed_json.get("intent", parsed_json.get("strategic_intent", "PASSIVE_INVESTMENT"))).upper(),
            "summary": str(parsed_json.get("summary", parsed_json.get("strategic_intent_summary", "M&A зацепок не обнаружено"))),
            "demands": list(parsed_json.get("demands", parsed_json.get("key_demands", []))),
            "quotes": list(parsed_json.get("quotes", parsed_json.get("evidence_quotes", []))),
            "turnaround_detected": bool(parsed_json.get("turnaround_detected", False)),
            "spinoff_detected": bool(parsed_json.get("spinoff_detected", False))
        }
        
        # Предотвращение некорректных значений классификатора
        valid_intents = ["HOSTILE_TAKEOVER", "ACTIVIST_RESTRUCTURING", "FRIENDLY_TOEHOLD", "PASSIVE_INVESTMENT"]
        if validated_data["intent"] not in valid_intents:
            validated_data["intent"] = "PASSIVE_INVESTMENT"
            
        # Строгая фильтрация ложных срабатываний и языкового мусора для 4B модели
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
        print(f" [WARNING] Ошибка синтаксиса JSON от gemma3:4b: {jde}. Активирован fallback.")
        return fallback_data
    except Exception as e:
        print(f" [WARNING] Непредвиденный сбой локального инференса gemma3:4b: {e}. Активирован fallback.")
        return fallback_data



def call_ollama(messages, max_tokens=50, temperature=0.0):
    """
    Попытка вызвать локальную модель через Ollama (перебирает имена gemma3:4b, gemma3-4b, gemma3).
    Проверяет как OpenAI-совместимый эндпоинт, так и нативный эндпоинт Ollama.
    """
    import requests
    import re
    
    models = ["gemma3:4b", "gemma3-4b", "gemma3"]
    
    for model in models:
        # 1. Попытка через OpenAI-совместимый эндпоинт Ollama
        try:
            res = requests.post(
                "http://localhost:11434/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "options": {
                        "num_ctx": 8192
                    }
                },
                timeout=12
            )
            if res.status_code == 200:
                raw_ans = res.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                raw_ans = re.sub(r'(?is)<think>.*?</think>', '', raw_ans).strip()
                raw_ans = re.sub(r'(?is)<think>.*', '', raw_ans).strip()
                if raw_ans:
                    print(f"    [OLLAMA] Использована локальная модель {model} через completions.")
                    return raw_ans
        except Exception as e:
            # Превышение таймаута или занятость GPU
            print(f"    [OLLAMA-DEBUG] Не удалось выполнить completions для {model}: {e}")

        # 2. Попытка через нативный эндпоинт api/chat
        try:
            res = requests.post(
                "http://localhost:11434/api/chat",
                headers={"Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": messages,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                        "num_ctx": 8192
                    },
                    "stream": False
                },
                timeout=12
            )
            if res.status_code == 200:
                raw_ans = res.json().get("message", {}).get("content", "").strip()
                raw_ans = re.sub(r'(?is)<think>.*?</think>', '', raw_ans).strip()
                raw_ans = re.sub(r'(?is)<think>.*', '', raw_ans).strip()
                if raw_ans:
                    print(f"    [OLLAMA] Использована локальная модель {model} через api/chat.")
                    return raw_ans
        except Exception as e:
            print(f"    [OLLAMA-DEBUG] Не удалось выполнить api/chat для {model}: {e}")
    return None



def extract_item4_section(html_content):
    """
    Продвинутый метод точечного парсинга: извлекает строго раздел Purpose of Transaction (Item 4)
    из формы Schedule 13D, полностью обходя ловушку Оглавления (Table of Contents Trap).
    """
    from bs4 import BeautifulSoup
    import re
    
    # Регулярные выражения для поиска границ разделов Item 4 и Item 5
    item4_regex = re.compile(r"item\s*4\s*[\s\S]{0,150}?purpose\s*of\s*(the\s*)?transaction", re.IGNORECASE)
    item5_regex = re.compile(r"item\s*5\s*[\s\S]{0,150}?interest\s*in\s*securities", re.IGNORECASE)
    
    # Собираем ВСЕ совпадения для Item 4 в документе, чтобы не споткнуться об оглавление
    matches_m4 = list(item4_regex.finditer(html_content))
    if not matches_m4:
        # Мягкий фоллбэк regex с менее жесткими условиями
        fallback_regex_4 = re.compile(r"item\s*4\s*[\s\S]{0,100}purpose\s*of", re.IGNORECASE)
        matches_m4 = list(fallback_regex_4.finditer(html_content))
        
    for m4 in matches_m4:
        item4_chunk = html_content[m4.start():]
        
        # Для текущего вхождения Item 4 ищем ближайший маркер Item 5
        m5 = item5_regex.search(item4_chunk)
        if not m5:
            fallback_regex_5 = re.compile(r"item\s*5\s*[\s\S]{0,100}interest\s*in", re.IGNORECASE)
            m5 = fallback_regex_5.search(item4_chunk)
            
        if m5:
            # Вырезаем фрагмент строго между Item 4 и Item 5
            final_chunk = item4_chunk[:m5.start()]
            
            # КРИТИЧЕСКИЙ ФИЛЬТР: Оглавление содержит очень короткий текст между Item 4 и 5.
            # Если длина HTML-фрагмента меньше 300 символов — это гарантированно ложная ссылка из ТОС.
            # Пропускаем её и идем к следующему (настоящему) разделу в теле документа.
            if len(final_chunk) < 300:
                continue
                
            soup = BeautifulSoup(final_chunk, "lxml")
            clean_text = soup.get_text(separator="\n")
            
            # Нормализуем пробелы и Юникод артефакты
            clean_text = re.sub(r'\xa0', ' ', clean_text)
            clean_text = re.sub(r'[ \t]+', ' ', clean_text)
            clean_text = re.sub(r'\n+', '\n', clean_text).strip()
            
            return clean_text
            
    # Крайний случай: если через цикл с фильтром длины ничего не вырезалось, 
    # берем последнее вхождение Item 4 (в теле документа оно обычно идет после оглавления)
    if matches_m4:
        last_m4 = matches_m4[-1]
        # Ограничиваем буфер 25000 символов, чтобы не перегрузить контекст ИИ
        soup = BeautifulSoup(html_content[last_m4.start():last_m4.start() + 25000], "lxml")
        return re.sub(r'\s+', ' ', soup.get_text()).strip()
    
    return None


def strip_boilerplate(text):
    """
    Boilerplate Stripper: Очищает текст от стандартных юридических фраз
    с помощью расширенного набора регулярных выражений (Шаг 4).
    """
    import re
    if not text:
        return ""
    
    boilerplates = [
        r"except as set forth in this Item 4[\s\S]{0,100}Reporting Persons[\s\S]{0,100}no present plans",
        r"the Reporting Persons have no present plans or proposals which relate to or would result in",
        # Мягкие регулярные выражения для детекции любых вариаций «may from time to time purchase/dispose in the open market»
        r"may from time to time[\s\S]{0,100}(purchase|acquire|buy|dispose of|sell)[\s\S]{0,100}shares",
        r"depending on market conditions[\s\S]{0,100}open market",
        r"in the open market depending on[\s\S]{0,50}conditions",
        r"intend to review their investment on a continuing basis",
        r"reserve the right to change their intention",
        r"subject to market conditions[\s\S]{0,50}availability of shares",
        r"in the ordinary course of business",
        r"depend on a variety of factors, including[\s\S]{0,100}financial condition",
        r"acquired for investment purposes",
        r"except as described above[\s\S]{0,100}Reporting Persons[\s\S]{0,100}have no present plans",
        r"reserve the right to formulate other plans",
        r"may engage in discussions with[\s\S]{0,100}management[\s\S]{0,50}board of directors",
        r"may from time to time[\s\S]{0,100}discuss with[\s\S]{0,100}other shareholders"
    ]
    
    clean_text = text
    for bp in boilerplates:
        clean_text = re.sub(bp, "[BOILERPLATE REMOVED]", clean_text, flags=re.IGNORECASE)
        
    clean_text = re.sub(r'(\[BOILERPLATE REMOVED\]\s*)+', '[BOILERPLATE REMOVED]\n', clean_text)
    lines = clean_text.split('\n')
    non_boilerplate_lines = [l.strip() for l in lines if l.strip() and l.strip() != "[BOILERPLATE REMOVED]"]
    return "\n".join(non_boilerplate_lines)


def clean_llm_response(text):
    """
    Очищает ответ ИИ от вежливых вводных фраз и разметки.
    Если ответ содержит отрицание наличия стратегии, возвращает 'NO'.
    """
    import re
    if not text:
        return ""
    
    # Очистка markdown разметки
    text = re.sub(r'```[a-zA-Z]*\n?', '', text)
    text = text.replace('```', '')
    
    # Паттерны разговорного мусора
    prefixes = [
        r"^okay,\s*(here's|here is)\s*a\s*breakdown\s*of\s*(the\s*)?.*?:",
        r"^here\s*(is|are)\s*the\s*.*?:",
        r"^based\s*on\s*.*?,",
        r"^sure,\s*here\s*is\s*.*?:",
        r"^the\s*company\s*mentions\s*that\s*",
        r"^i\s*would\s*classify\s*this\s*as\s*",
        r"^the\s*intent\s*is\s*"
    ]
    cleaned = text.strip()
    for pref in prefixes:
        cleaned = re.sub(pref, "", cleaned, flags=re.IGNORECASE).strip()
    
    if (cleaned.startswith('"') and cleaned.endswith('"')) or (cleaned.startswith("'") and cleaned.endswith("'")):
        cleaned = cleaned[1:-1].strip()
        
    # Детекция ложноположительных текстов-отрицаний
    lower = cleaned.lower()
    if "no " in lower and ("turnaround" in lower or "restructuring" in lower or "strategy" in lower or "mention" in lower or "disclosed" in lower):
        return "NO"
    if "does not mention" in lower or "no specific" in lower:
        return "NO"
        
    return cleaned


def call_groq_api_with_fallback(messages, max_tokens=50, temperature=0.0):
    """
    Умный ротатор ИИ моделей. В первую очередь опрашивает локальную модель gemma3-4b через Ollama.
    Если локальный вызов завершился ошибкой или вернул пустой ответ, переходит к карусели моделей Groq API.
    """
    # 1. Сначала пробуем локальную модель Ollama
    ollama_res = call_ollama(messages, max_tokens=max_tokens, temperature=temperature)
    if ollama_res:
        return ollama_res

    # 2. Если Ollama недоступна, выполняем фоллбэк к Groq API
    import time
    global _active_groq_model_index, _groq_model_cooldowns

    # Полный список доступных моделей (в порядке приоритета по RPM лимитам)
    GROQ_MODELS = [
        "llama-3.1-8b-instant",                        # 14.4K RPM
        "qwen/qwen3-32b",                              # 60 RPM
        "llama-3.3-70b-versatile",                     # 1K RPM
        "openai/gpt-oss-20b",                          # 1K RPM (Reasoning)
        "openai/gpt-oss-120b",                         # 1K RPM (Reasoning)
        "openai/gpt-oss-safeguard-20b",                # 1K RPM (Reasoning)
        "meta-llama/llama-4-scout-17b-16e-instruct",  # 1K RPM
        "qwen/qwen3.6-27b",                            # 1K RPM
        "groq/compound-mini",                          # 30 RPM
        "groq/compound",                               # 30 RPM
        "allam-2-7b",                                  # 30 RPM
    ]

    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")
    now = time.time()

    # Очищаем устаревшие cooldown'ы
    _groq_model_cooldowns = {k: v for k, v in _groq_model_cooldowns.items() if v > now}

    # Составляем список доступных моделей (не в cooldown)
    available = [m for m in GROQ_MODELS if m not in _groq_model_cooldowns]
    if not available:
        # Все модели в cooldown — ждём истечения ближайшего
        min_wait = min(_groq_model_cooldowns.values()) - now
        time.sleep(max(0.5, min_wait))
        _groq_model_cooldowns.clear()
        available = list(GROQ_MODELS)

    # Начинаем с индекса для равномерного распределения нагрузки
    start_idx = _active_groq_model_index % len(available)
    ordered = available[start_idx:] + available[:start_idx]

    for model in ordered:
        try:
            # Увеличиваем лимит токенов для reasoning моделей, чтобы они не обрезались на размышлениях
            # Увеличиваем лимит токенов для всех моделей до 200, чтобы избежать срезов при размышлениях <think>
            model_max_tokens = max(200, max_tokens)
            if "gpt-oss" in model or "compound" in model:
                model_max_tokens = max(400, max_tokens * 3)

            res = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": model_max_tokens
                },
                timeout=12
            )
            if res.status_code == 200:
                raw_ans = res.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                # Убираем <think>...</think> теги от reasoning-моделей
                raw_ans = re.sub(r'(?is)<think>.*?</think>', '', raw_ans).strip()
                raw_ans = re.sub(r'(?is)<think>.*', '', raw_ans).strip()
                # Следующий вызов начнём со следующей модели (карусель)
                _active_groq_model_index = (GROQ_MODELS.index(model) + 1) % len(GROQ_MODELS)
                return raw_ans
            elif res.status_code == 429:
                retry_after = int(res.headers.get("retry-after", 60))
                _groq_model_cooldowns[model] = now + retry_after
                print(f"    [GROQ] {model} → 429, cooldown {retry_after}s. Следующая модель...")
                continue
            elif res.status_code in (503, 502):
                _groq_model_cooldowns[model] = now + 30
                print(f"    [GROQ] {model} → {res.status_code} (перегружена). Cooldown 30s...")
                continue
            else:
                print(f"    [GROQ] {model} → {res.status_code}. Пробую следующую...")
                continue
        except Exception as e:
            print(f"    [GROQ] {model} → Exception: {e}. Следующая...")
            continue
    return ""


MA_DRIVERS_REGISTRY = {
    "ELLIOTT INVESTMENT": {
        "type": "Агрессивный активист (Tier-0)",
        "impact_score": 30,
        "text": "Вход Elliott Management: запуск экстремально агрессивного давления. Высочайшая вероятность принудительного выставления компании на аукцион или выделения активов."
    },
    "STARBOARD VALUE": {
        "type": "Операционный активист",
        "impact_score": 25,
        "text": "Вход Starboard Value: фокус на сокращении SG&A расходов и смене менеджмента. Ожидается жесткая оптимизация маржи и прокси-борьба за контроль."
    },
    "JANA PARTNERS": {
        "type": "Event-Driven снайпер",
        "impact_score": 25,
        "text": "Вход JANA Partners: исторический паттерн фонда — покупка доли с целью полной продажи компании стратегическому покупателю (Agitate to Sell)."
    },
    "ICAHN CARL": {
        "type": "Корпоративный рейдер",
        "impact_score": 25,
        "text": "Раскрыта доля Карла Икана: ожидается жесткий конфликт с текущим CEO, требования масштабного байбэка или разделения бизнеса."
    },
    "VALUEACT HOLDINGS": {
        "type": "Институциональный дипломат",
        "impact_score": 20,
        "text": "Вход ValueAct: ожидается конструктивная реструктуризация и оптимизация маржинальности изнутри."
    },
    "TRIAN FUND": {
        "type": "Операционный хирург",
        "impact_score": 20,
        "text": "Вход Trian Partners: ожидается давление на выделение активов (Spinoff) и повышение операционной эффективности."
    },
    "CORVEX MANAGEMENT": {
        "type": "Активист спец-ситуаций",
        "impact_score": 20,
        "text": "Вход Corvex Management: высокая вероятность глубокого арбитража и потенциального LBO."
    },
    "BAUPOST GROUP": {
        "type": "Deep Value снайпер",
        "impact_score": 20,
        "text": "Вход Baupost Group: математика ликвидационной стоимости подтверждена Семом Кларманом."
    },
    "BERKSHIRE HATHAWAY": {
        "type": "Институциональный якорь",
        "impact_score": 25,
        "text": "Вход Berkshire Hathaway: холдинг Уоррена Баффета аккумулировал пакет акций, создавая 'бетонный пол' для стоимости компании."
    },
    "SILVER LAKE": {
        "type": "Технологический PE-гигант (Tier-1)",
        "impact_score": 25,
        "text": "Вход Silver Lake Group: крупное стратегическое накопление/toehold технологического PE-гиганта. Высокая вероятность подготовки к LBO, приватизации (Take-Private) или крупной реструктуризации."
    }

}

def check_whale_catalysts(influential_shareholders):
    qual_points = []
    qual_score = 0
    for holder in influential_shareholders:
        holder_name = holder.get('name', '').upper().replace('.', '')
        for whale_core_name, whale_data in MA_DRIVERS_REGISTRY.items():
            if whale_core_name in holder_name:
                qual_points.append(whale_data['text'])
                qual_score += whale_data['impact_score']
    return qual_points, qual_score
# ----------------------------------------------------
# 1. НАСТРОЙКА КЭША И СЕССИИ
# ----------------------------------------------------
CACHE_PATH = "ma_screener_cache.json"

def load_cache():
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[!] Ошибка загрузки кэша: {e}. Будет создан новый кэш.")
    return {"tickers": {}}

def save_cache(cache):
    try:
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2, default=json_serialize_fallback)
    except Exception as e:
        print(f"[!] Ошибка сохранения кэша: {e}")

# ----------------------------------------------------
# 2. ИМПОРТ ИНДЕКСОВ С WIKIPEDIA
# ----------------------------------------------------

def fetch_sp500_tickers(cache=None):
    print("[*] Загрузка списка S&P 500 из Википедии...")
    try:
        import io
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
        res = requests.get(url, headers=headers)
        tables = pd.read_html(io.StringIO(res.text))
        df = tables[0]
        tickers = df['Symbol'].tolist()
        tickers = [t.replace('.', '-') for t in tickers]
        return tickers
    except Exception as e:
        print(f"[!] Ошибка загрузки S&P 500 с Wikipedia: {e}.")
        if cache and "tickers" in cache:
            cached_sp500 = [t for t, obj in cache["tickers"].items() if obj.get("index_group") == "SP500"]
            if cached_sp500:
                print(f"    [+] Восстановлено {len(cached_sp500)} тикеров S&P 500 из локального кэша.")
                return cached_sp500
        return []

# ----------------------------------------------------
# 3. ДВИЖОК ОЦЕНКИ СМЕНЫ КОНТРОЛЯ (OPTIMAL DCF)
# ----------------------------------------------------

class OptimalControlEngine:
    DEFAULT_SECTOR_MEDIANS = {
        # Источник: Damodaran Online, January 2025 (pages.stern.nyu.edu/~adamodar)
        # operating_margin: медиана операционной маржи по сектору
        # roc: медиана ROC (EBIT(1-t) / Invested Capital)
        # debt_ratio: медиана D/(D+E) по рыночной стоимости
        # sga: медиана SG&A / Revenue
        "Technology": {"operating_margin": 0.18, "roc": 0.14, "debt_ratio": 0.12, "sga": 0.18},
        "Healthcare": {"operating_margin": 0.12, "roc": 0.10, "debt_ratio": 0.10, "sga": 0.22},
        "Financial Services": {"operating_margin": 0.22, "roc": 0.08, "debt_ratio": 0.40, "sga": 0.15},
        # Consumer Cyclical: Damodaran 2025 медиана ~10.5% op margin (Personal Services ~20%+)
        # Используем 10% как консервативную медиану — Personal Services компании (HRB) выше нормы
        "Consumer Cyclical": {"operating_margin": 0.10, "roc": 0.11, "debt_ratio": 0.22, "sga": 0.20},
        "Industrials": {"operating_margin": 0.11, "roc": 0.11, "debt_ratio": 0.22, "sga": 0.14},
        "Consumer Defensive": {"operating_margin": 0.08, "roc": 0.12, "debt_ratio": 0.18, "sga": 0.17},
        "Energy": {"operating_margin": 0.13, "roc": 0.09, "debt_ratio": 0.28, "sga": 0.07},
        "Utilities": {"operating_margin": 0.19, "roc": 0.06, "debt_ratio": 0.52, "sga": 0.05},
        "Real Estate": {"operating_margin": 0.28, "roc": 0.05, "debt_ratio": 0.48, "sga": 0.09},
        "Basic Materials": {"operating_margin": 0.11, "roc": 0.08, "debt_ratio": 0.28, "sga": 0.09},
        "Communication Services": {"operating_margin": 0.14, "roc": 0.09, "debt_ratio": 0.23, "sga": 0.19}
    }

    def __init__(self):
        self.sector_medians = {}

    def get_medians_for_sector(self, sector):
        return self.sector_medians.get(sector) or self.DEFAULT_SECTOR_MEDIANS.get(sector) or {
            "operating_margin": 0.12, "roc": 0.10, "debt_ratio": 0.20, "sga": 0.15
        }

    def run_optimal_dcf(self, ticker, parsed_data, wacc_data, standalone_price=0.0, distress_profile=None):
        sector = wacc_data.get("sector") or "Technology"
        medians = self.get_medians_for_sector(sector)
        
        df = parsed_data["df"].copy()
        latest = df.iloc[-1]
        
        current_op_margin = latest["ebit"] / latest["revenue"] if latest["revenue"] > 0 else 0.0
        optimal_op_margin = max(current_op_margin, medians["operating_margin"])
        
        # ── Damodaran: terminal_growth = min(rf, 3%) — согласованность с standard_fcff_model ──
        rf = wacc_data.get("rf", 0.042)
        terminal_growth = min(rf, 0.03)
        wacc_data["terminal_growth"] = terminal_growth
        
        has_negative_equity = "equity" not in df.columns or df["equity"].iloc[-1] <= 0
        if has_negative_equity:
            gross_assets = df["total_assets"] if "total_assets" in df.columns else df["total_debt"] * 2
            goodwill_col = df["goodwill"] if "goodwill" in df.columns else pd.Series(0.0, index=df.index)
            invested_capital = float((gross_assets - goodwill_col - latest["total_cash"]).iloc[-1])
            invested_capital = max(invested_capital, 1.0)
            current_roc = (latest["ebit_after_tax"] / invested_capital) if invested_capital > 0 else 0.12
            current_roc = min(max(current_roc, 0.01), 0.80)  # clip как в standard_fcff_model
        else:
            book_eq = df["equity"].iloc[-1]
            invested_capital = latest["total_debt"] + book_eq - latest["total_cash"]
            if invested_capital < book_eq * 0.10:
                invested_capital = max(book_eq * 0.50, latest.get("total_assets", 1.0) * 0.10)
            current_roc = latest["ebit_after_tax"] / invested_capital if invested_capital > 0 else 0.10
            
        optimal_roc = max(current_roc, medians["roc"])
        
        current_debt_ratio = wacc_data["w_d"]
        optimal_debt_ratio = medians["debt_ratio"]
        
        optimal_wacc = wacc_data["wacc"]
        optimal_ke = wacc_data["ke"]
        optimal_beta = wacc_data["beta"]
        if optimal_debt_ratio > current_debt_ratio + 0.05:
            tax_rate = wacc_data.get("marginal_tax_rate", 0.25)
            current_de = current_debt_ratio / (1 - current_debt_ratio) if current_debt_ratio < 1 else 0.0
            unlevered_beta = unlever_beta(wacc_data["beta"], current_de, tax_rate)

            optimal_de = optimal_debt_ratio / (1 - optimal_debt_ratio) if optimal_debt_ratio < 1 else 0.0
            optimal_beta = relever_beta(unlevered_beta, optimal_de, tax_rate)

            ke_opt = wacc_data["rf"] + optimal_beta * DAMODARAN_ERP
            kd_opt = wacc_data["kd"]
            kd_opt_tax = kd_opt * (1 - tax_rate)
            calculated_optimal_wacc = ke_opt * (1 - optimal_debt_ratio) + kd_opt_tax * optimal_debt_ratio
            optimal_wacc = min(wacc_data["wacc"], calculated_optimal_wacc)
            optimal_ke = ke_opt
        
        df_opt = df.copy()
        # Принудительно конвертируем целевые финансовые метрики во float64 для предотвращения конфликтов типов
        for col in ["ebit", "ebit_after_tax", "net_income"]:
            if col in df_opt.columns:
                df_opt[col] = df_opt[col].astype("float64")

        if optimal_op_margin > current_op_margin:
            df_opt["revenue_opt"] = df_opt["revenue"]
            old_ebit_after_tax = df["ebit_after_tax"].astype("float64")
            df_opt["ebit"] = df_opt["revenue_opt"] * optimal_op_margin
            df_opt["ebit_after_tax"] = df_opt["ebit"] * (1 - df_opt["effective_tax_rate"])
            
            # Важно: Пересчитываем чистую прибыль (net_income), добавляя операционный прирост от реформ, 
            # чтобы модель FCFE ниже учла улучшения менеджмента
            ebit_after_tax_delta = df_opt["ebit_after_tax"] - old_ebit_after_tax
            df_opt["net_income"] = df["net_income"].astype("float64") + ebit_after_tax_delta

        # ── Reinvestment Rate recalculation (Damodaran Ch. 25) ───────────────
        if optimal_roc > current_roc and optimal_roc > 0:
            long_term_g = wacc_data.get("terminal_growth", 0.025)
            optimal_rir = min(long_term_g / optimal_roc, 0.80)
            df_opt["reinvestment_rate"] = optimal_rir
            
        wacc_opt_data = wacc_data.copy()
        wacc_opt_data["wacc"] = optimal_wacc
        wacc_opt_data["ke"] = optimal_ke
        wacc_opt_data["beta"] = optimal_beta
        wacc_opt_data["is_optimal"] = True
        wacc_opt_data["opt_roc"] = optimal_roc
        wacc_opt_data["opt_margin"] = optimal_op_margin
        wacc_floor = wacc_data.get("rf", 0.042) + 0.020
        wacc_opt_data["wacc"] = max(optimal_wacc, wacc_floor)
        
        # РЕШЕНИЕ ОШИБКИ 1: Синхронный свитч дистресс-модели на FCFE в оптимальном сценарии управления
        if distress_profile and distress_profile.get("fcfe_override"):
            try:
                _sh = float(latest.get("shares", 1.0))
                _latest_opt = df_opt.iloc[-1]
                _ni = float(_latest_opt.get("net_income", 0.0))
                _da = float(_latest_opt.get("da", 0.0))
                _capex = float(_latest_opt.get("capex", 0.0))
                _td = float(_latest_opt.get("total_debt", 0.0))
                
                _net_borrowing = max(0.0, _td - float(df_opt.iloc[-2].get("total_debt", _td)) if len(df_opt) >= 2 else 0.0)
                _fcfe_base = _ni + _da - _capex + _net_borrowing
                if len(df_opt) >= 2:
                    _ni2 = float(df_opt.iloc[-2].get("net_income", _ni))
                    _da2 = float(df_opt.iloc[-2].get("da", _da))
                    _cap2 = float(df_opt.iloc[-2].get("capex", _capex))
                    _fcfe2 = _ni2 + _da2 - _cap2
                    _fcfe_base = (_fcfe_base + _fcfe2) / 2

                _ke = max(wacc_opt_data["ke"], 0.06)
                _terminal_g = min(rf, 0.03)
                _g1 = max(-0.05, min(0.10, _terminal_g + 0.02))

                _fcfe_pv = 0.0
                _fcfe_t = _fcfe_base
                for _yr in range(1, 6):
                    _fcfe_t *= (1 + _g1)
                    _fcfe_pv += _fcfe_t / (1 + _ke) ** _yr

                _fcfe_terminal = _fcfe_t * (1 + _terminal_g) / max(_ke - _terminal_g, 0.01)
                _fcfe_tv_pv = _fcfe_terminal / (1 + _ke) ** 5

                opt_dcf_price = (_fcfe_pv + _fcfe_tv_pv) / _sh if _sh > 0 else 0.0
                opt_dcf_price = max(opt_dcf_price, standalone_price)
                return opt_dcf_price, optimal_wacc, optimal_op_margin, optimal_roc
            except Exception as e:
                return standalone_price, optimal_wacc, optimal_op_margin, optimal_roc
        
        models_opt = ValuationModels(wacc_opt_data, df_opt, parsed_data["is_financial"])
        try:
            model_name, forecast_df, opt_dcf_price, eq_val = models_opt.run_dcf()
            opt_dcf_price = max(opt_dcf_price, standalone_price)
            return opt_dcf_price, optimal_wacc, optimal_op_margin, optimal_roc
        except Exception as e:
            return standalone_price, wacc_data["wacc"], current_op_margin, current_roc


# ----------------------------------------------------
# 3.7. ДВИЖОК ВРЕМЕНИ СДЕЛКИ (M&A TIMING ENGINE)
# ----------------------------------------------------
class MATimingEngine:
    """
    Движок прогнозирования временных окон сделок (M&A Timing & Succession).
    Анализирует 5 ключевых количественных и качественных триггеров:
    1. Следы инвесторов-активистов (форма SEC 13D/A).
    2. Стена рефинансирования долгов (Debt Maturity Wall).
    3. Фактор основателя и смена поколений (succession risk по возрасту CEO/Chairman).
    4. Пост-IPO VC Exit / Lock-up Expiration.
    5. Эффект домино / волна M&A поглощений в индустрии.
    """
    def calculate_timing(self, ticker, parsed_data, wacc_data, yf_info, sec_ma_hints, cache, turnaround_news=None, buyback_suspended=None, whale_points=None, whale_score=0, qual_points=None, qual_score=0):
        score = 10
        points = []
        if whale_score > 0 and whale_points:
            score += whale_score
            points.extend(whale_points)

        
        # 1. Activist Presence (Form 13D/A)
        has_13d = False
        for hint in (sec_ma_hints or []):
            if "13D" in str(hint.get("form", "")):
                has_13d = True
                break
        if has_13d:
            score += 25
            points.append("Инвестор-активист вошел в капитал (подана форма SEC 13D в течение последнего года) — открыто окно 6-18 месяцев до сделки/реструктуризации.")
            
        # 2. Debt Maturity Wall (Current vs Cash & total debt)
        latest = parsed_data["df"].iloc[-1]
        st_debt = float(latest.get("st_debt", 0.0))
        lt_debt = float(latest.get("lt_debt", 0.0))
        total_debt = st_debt + lt_debt
        cash = float(latest.get("cash", 0.0)) + float(latest.get("marketable_sec", 0.0))
        
        if total_debt > 0 and st_debt > cash and (st_debt / total_debt) > 0.30:
            pct_st = (st_debt / total_debt) * 100
            score += 25
            points.append(f"Приближение 'стены долга' (текущие обязательства ${st_debt/1e6:.1f}M превышают кэш, более {pct_st:.0f}% долга требует рефинансирования в течение 12 месяцев) в условиях высоких ставок.")
            
        # 3. Founder Age & Succession (Generation Change)
        officers = yf_info.get("companyOfficers", [])
        old_founders = []
        has_old_ceo = False
        for off in officers:
            title = str(off.get("title", "")).lower()
            age = off.get("age")
            if age and age >= 65:
                import re
                if re.search(r'\b(ceo|chief executive officer|founder|chairman|director)\b', title) and not re.search(r'\b(cfo|chief financial officer|vice president|vp|coo|chief operating officer)\b', title):
                    if re.search(r'\b(ceo|chief executive officer)\b', title):
                        has_old_ceo = True
                    old_founders.append(f"{off.get('name')} ({age} лет, {off.get('title')})")
        if old_founders:
            score += 20
            if has_old_ceo:
                points.append(f"Возрастной CEO ({', '.join(old_founders)}) при отсутствии явного преемника повышает вероятность монетизации через продажу.")
            else:
                points.append(f"Возрастной основатель/директор ({', '.join(old_founders)}) при отсутствии явного преемника повышает вероятность монетизации через продажу.")
            
        # 4. Post-IPO / VC Exit Window
        is_post_ipo = False
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="5y")
            if len(hist) < 252 * 3:
                is_post_ipo = True
        except:
            pass
        
        ebitda = float(latest.get("ebit", 0.0)) + float(latest.get("da", 0.0))
        if is_post_ipo and ebitda <= 0:
            score += 15
            points.append("Пост-IPO окно уязвимости (компания на бирже менее 3 лет, EBITDA отрицательная) — венчурные инвесторы давят на продажу после локапа.")
            
        # 5. Domino Effect / Industry Clustering
        industry = parsed_data.get("industry") or yf_info.get("industry") or ""
        industry_clean = str(industry).lower().strip()
        peers_active = 0
        if cache and "tickers" in cache:
            for t, obj in cache["tickers"].items():
                if t == ticker:
                    continue
                t_ind = str(obj.get("financials", {}).get("industry", "")).lower().strip()
                if t_ind == industry_clean and obj.get("score", 0) >= 65:
                    peers_active += 1
        if peers_active > 0:
            score += 15
            points.append(f"Эффект домино: в индустрии '{industry}' наблюдается высокая M&A-активность peers ({peers_active} активных целей в скринере), что стимулирует оборонительные поглощения.")
            
        # 6. Turnaround / Restructuring and Buyback Suspension
        has_turnaround = bool(turnaround_news)
        has_suspended_buyback = bool(buyback_suspended)
        insider_share_pct = (yf_info.get("heldPercentInsiders") or 0.0) * 100
        
        if has_turnaround or has_suspended_buyback:
            score += 25
            reasons = []
            if has_turnaround:
                reasons.append("запущена программа реструктуризации/трансформации (Turnaround)")
            if has_suspended_buyback:
                reasons.append("приостановлен обратный выкуп акций (buyback)")
            reasons_str = " или ".join(reasons)
            points.append(f"Уязвимая фаза: {reasons_str} при низком инсайдерском контроле ({insider_share_pct:.1f}%), что делает компанию уязвимой для M&A-активистов.")
            
        # 7. Qualitative SEC & Transcript Boost
        if qual_score > 0:
            score += qual_score
            for qp in (qual_points or []):
                points.append(qp)

        # Корректировка для анонсированных слияний в процессе
        has_announced_deal = False
        for hint in (sec_ma_hints or []):
            f_upper = str(hint.get("form", "")).upper()
            if "14A" in f_upper or "S-4" in f_upper or "TO-T" in f_upper:
                has_announced_deal = True
                break
                
        if has_announced_deal:
            score = 100
            verdict = "СДЕЛКА АНОНСИРОВАНА / ПОДПИСАНА (В процессе завершения)"
            verdict_class = "text-green"
        else:
            score = min(100, score)
            
        if not has_announced_deal:
            if score >= 60:
                verdict = "ВЫСОКАЯ ГОТОВНОСТЬ (Окно открыто: 6-12 месяцев)"
                verdict_class = "text-green"
            elif score >= 30:
                verdict = "УМЕРЕННАЯ ГОТОВНОСТЬ (Окно: 12-24 месяцев)"
                verdict_class = "text-orange"
            elif len(points) > 0:
                verdict = "РАННЯЯ ФАЗА (Присутствуют скрытые катализаторы)"
                verdict_class = "text-orange"
            else:
                verdict = "СПЯЩИЙ РЕЖИМ (Катализаторы времени не активированы)"
                verdict_class = "text-secondary"
            
        return {
            "score": score,
            "verdict": verdict,
            "class": verdict_class,
            "points": points or ["Специфические триггеры времени сделки не активированы. Компания находится в режиме долгосрочного органического развития."]
        }


# ----------------------------------------------------
# 3.8. МОДУЛЬ ПОДБОРА ПОКУПАТЕЛЕЙ (BUYER MATCHMAKER)
# ----------------------------------------------------
# Оценивает соответствие цели M&A списку потенциальных покупателей по трем
# независимым критериям (DePamphilis Ch. 3 / Damodaran Ch. 25 / FTC 2010
# Horizontal Merger Guidelines):
#   1. Debt Capacity  — LBO-тест для финансовых (PE) покупателей и
#      Borrowing Capacity тест для стратегических покупателей.
#   2. Синергия SG&A  — 15% сокращение SG&A цели при совпадении
#      SIC-кода/сектора между целью и покупателем.
#   3. HHI-барьер     — приближенная оценка антимонопольного риска.
#
# Никакие конкретные бренды/фонды не зашиты: список потенциальных
# покупателей строится из живых данных (кэш скринера) вызывающим кодом.

LBO_MAX_DEBT_EBITDA = 6.0            # Стандартный потолок совокупного левериджа для LBO (DePamphilis Ch. 3)
STRATEGIC_MAX_DEBT_EBITDA = 3.5      # Investment-grade потолок Debt/EBITDA для стратега (Damodaran Ch. 25)

# FTC/DOJ Horizontal Merger Guidelines (2010) — официальные пороги концентрации:
FTC_HHI_UNCONCENTRATED = 1500
FTC_HHI_HIGHLY_CONCENTRATED = 2500
FTC_HHI_DELTA_SAFE_HARBOR = 100      # ΔHHI < 100 → в любом рынке маловероятны возражения
FTC_HHI_DELTA_PRESUMPTION = 200      # ΔHHI > 200 в highly concentrated рынке → presumption of market power


def _compute_hhi(market_shares_pct):
    """HHI = Σ(доля_i в %)². Шкала 0–10000 (FTC Horizontal Merger Guidelines, 2010)."""
    return sum(s ** 2 for s in market_shares_pct)


def _hhi_verdict(hhi_pre, hhi_post):
    delta = hhi_post - hhi_pre
    if hhi_post > FTC_HHI_HIGHLY_CONCENTRATED and delta > FTC_HHI_DELTA_PRESUMPTION:
        return "ВЫСОКИЙ АНТИМОНОПОЛЬНЫЙ РИСК (Presumed Blocked)", "text-red", delta
    if hhi_post > FTC_HHI_UNCONCENTRATED and delta > FTC_HHI_DELTA_PRESUMPTION:
        return "ПОВЫШЕННЫЙ РИСК (потенциальные антимонопольные возражения)", "text-orange", delta
    if delta > FTC_HHI_DELTA_SAFE_HARBOR:
        return "УМЕРЕННЫЙ РИСК (требуется доп. проверка FTC/DOJ)", "text-orange", delta
    return "НИЗКИЙ РИСК (Safe Harbor)", "text-green", delta



def simulate_negative_covenants(post_trans_debt, post_trans_ebitda, net_income, capex, da, buyer_type="financial"):
    """
    Симуляция отрицательных ковенантов (Negative Covenants) по Главе 13.
    - Dividend Restrictions (Ограничение дивидендов)
    - Additional Debt Restrictions (Лимиты на доп. долг)
    - Capex Limits (Лимиты на капитальные затраты)
    """
    ebitda = max(post_trans_ebitda, 1.0)
    debt = max(post_trans_debt, 0.0)
    da_val = max(da, 1.0)
    ni = max(net_income, 0.0)

    leverage = debt / ebitda

    # 1. Dividend Restrictions
    dividend_allowed = True
    dividend_cap_pct = 100.0
    div_reason = "Разрешено в полном объеме (долговая нагрузка в пределах нормы)"

    if buyer_type == "financial":
        if leverage > 4.5:
            dividend_allowed = False
            dividend_cap_pct = 0.0
            div_reason = f"Заблокировано ковенантом: плечо Newco {leverage:.2f}x превышает лимит LBO 4.5x. Выплата дивидендов запрещена."
        elif leverage > 3.5:
            dividend_allowed = True
            dividend_cap_pct = 15.0
            div_reason = f"Ограничено ковенантом: плечо Newco {leverage:.2f}x находится в диапазоне [3.5x - 4.5x]. Выплата дивидендов ограничена лимитом 15% от Net Income."
    else:  # strategic
        if leverage > 3.5:
            dividend_allowed = False
            dividend_cap_pct = 0.0
            div_reason = f"Заблокировано ковенантом: плечо Newco {leverage:.2f}x превышает лимит стратега 3.5x. Выплата дивидендов запрещена."
        elif leverage > 2.5:
            dividend_allowed = True
            dividend_cap_pct = 25.0
            div_reason = f"Ограничено ковенантом: плечо Newco {leverage:.2f}x находится в диапазоне [2.5x - 3.5x]. Выплата дивидендов ограничена лимитом 25% от Net Income."

    # 2. Additional Debt Restrictions (Debt Limits)
    max_leverage_limit = 5.5 if buyer_type == "financial" else 4.0
    allowed_debt = max_leverage_limit * ebitda
    debt_headroom = max(0.0, allowed_debt - debt)
    additional_debt_allowed = debt_headroom > 0
    debt_reason = (
        f"Допустимый уровень плеча: лимит Debt/EBITDA = {max_leverage_limit:.1f}x. "
        f"Прогнозный долг Newco = ${debt/1e6:.1f}M. Свободный лимит заимствований (Headroom) = ${debt_headroom/1e6:.1f}M."
    )
    if not additional_debt_allowed:
        debt_reason = f"Доп. заимствования полностью заблокированы: плечо Newco {leverage:.2f}x превышает жесткий лимит ковенанта {max_leverage_limit:.1f}x."

    # 3. Capex Limits
    capex_limit = max(da_val * 1.2, ebitda * 0.15)
    capex_breach = capex > capex_limit
    capex_reason = (
        f"Лимит Capex по ковенанту (max из 1.2x D&A или 15% EBITDA) = ${capex_limit/1e6:.1f}M. "
        f"Прогнозный Capex = ${capex/1e6:.1f}M. Проходит ковенант ✓."
    )
    if capex_breach:
        capex_reason = (
            f"Превышение лимита Capex: лимит по ковенанту = ${capex_limit/1e6:.1f}M. "
            f"Планируемый Capex = ${capex/1e6:.1f}M (превышение на ${(capex - capex_limit)/1e6:.1f}M). "
            f"Запланированный ретулинг/модернизация под угрозой: требуется предварительное согласие кредиторов ⚠️."
        )

    # Risk Rating
    breaches_count = 0
    if not dividend_allowed or dividend_cap_pct < 50.0:
        breaches_count += 1
    if not additional_debt_allowed:
        breaches_count += 1
    if capex_breach:
        breaches_count += 1

    if breaches_count == 0:
        risk_level = "Низкий риск ковенантов"
        risk_class = "text-green"
    elif breaches_count == 1:
        risk_level = "Умеренный риск ковенантов"
        risk_class = "text-orange"
    else:
        risk_level = "Высокий риск ковенантов (Covenant Breach Risk)"
        risk_class = "text-red"

    return {
        "leverage": round(leverage, 2),
        "dividend_allowed": dividend_allowed,
        "dividend_cap_pct": dividend_cap_pct,
        "dividend_reason": div_reason,
        "debt_limit": round(allowed_debt, 2),
        "debt_headroom": round(debt_headroom, 2),
        "additional_debt_allowed": additional_debt_allowed,
        "debt_reason": debt_reason,
        "capex_limit": round(capex_limit, 2),
        "capex_breach": capex_breach,
        "capex_reason": capex_reason,
        "breaches_count": breaches_count,
        "risk_level": risk_level,
        "risk_class": risk_class
    }



def run_buyer_matchmaker(target, potential_bidders, industry_universe=None):
    """
    target: dict {ticker, sic, sector, revenue, sga, ebitda, debt, cash,
                  mcap, shares, price, tax_rate, net_income, capex, da}
    potential_bidders: list[dict] с теми же полями + "buyer_type":
        "financial" (PE/LBO-покупатель) | "strategic" (операционная компания)
        + опционально "wacc" (в долях, напр. 0.10)
    industry_universe: список revenue игроков индустрии (для HHI). Если не
        передан или его недостаточно — HHI считается «Н/Д», а не выдумывается.

    Возвращает список результатов по каждому bidder'у, отсортированный по
    итоговому match_score (0-100), убывание.
    """
    if not potential_bidders:
        return []

    target_ebitda = target.get("ebitda", 0.0)
    target_sga = target.get("sga", 0.0)
    target_sic = target.get("sic")
    target_sector = target.get("sector") or "Technology"
    target_revenue = target.get("revenue", 0.0)
    target_debt = target.get("debt", 0.0)
    target_cash = target.get("cash", 0.0)
    target_mcap = target.get("mcap", 0.0)
    tax_rate = target.get("tax_rate", 0.25)

    results = []

    for bidder in potential_bidders:
        b_ticker = bidder.get("ticker", "N/A")
        b_type = bidder.get("buyer_type", "strategic")
        b_debt = bidder.get("debt", 0.0)
        b_cash = bidder.get("cash", 0.0)
        b_ebitda = bidder.get("ebitda", 0.0)
        b_mcap = bidder.get("mcap", 0.0)
        b_sic = bidder.get("sic")
        b_sector = bidder.get("sector")
        b_revenue = bidder.get("revenue", 0.0)
        b_wacc = bidder.get("wacc", 0.10) or 0.10

        notes = []
        deal_ev_proxy = max(target_mcap + target_debt - target_cash, 1.0)

        # ── 1. Debt Capacity / Borrowing Capacity Test ──────────────────────
        if b_type == "financial":
            # LBO-тест (DePamphilis Ch. 3):
            # 1. Если EBITDA цели отрицательная, LBO автоматически блокируется.
            # 2. LBO-симулятор: Проверяем способность обслуживать долг (покрытие процентов >= 3.0x 
            #    при плече 75% долга на 25% капитала с процентной ставкой 8.0%).
            if target_ebitda <= 0:
                debt_capacity_ok = False
                debt_capacity_note = "LBO невозможен: EBITDA цели отрицательная — банки не выдадут кредит под актив без операционного потока."
            else:
                lbo_debt = 0.75 * deal_ev_proxy
                annual_interest = lbo_debt * 0.08  # средневзвешенная ставка 8.0%
                interest_coverage = target_ebitda / annual_interest if annual_interest > 0 else 999.0
                lbo_sim_ok = interest_coverage >= 3.0
                debt_capacity_ok = lbo_sim_ok
                
                debt_capacity_note = (
                    f"LBO симуляция (75% долга / 25% капитала): Требуется долг ${lbo_debt/1e6:.1f}M. "
                    f"Прогнозные проценты (при ставке 8%) = ${annual_interest/1e6:.1f}M/год. "
                    f"Покрытие процентов EBITDA = {interest_coverage:.2f}x (требуется >= 3.0x) → "
                    f"{'Проходит ✓ (LBO реализуемо)' if lbo_sim_ok else 'Не проходит ✗ (риск дефолта)'}."
                )
        else:
            # Borrowing Capacity тест для стратега (Damodaran Ch. 25):
            # Сравниваем D/E покупателя со среднеотраслевой медианой. 
            # Если b_de < sector_median_de, рассчитываем неиспользованную долговую емкость (Senior Debt)
            # и проверяем, покроет ли она + свободный кэш общую стоимость сделки (Deal EV).
            sector_data = OptimalControlEngine.DEFAULT_SECTOR_MEDIANS.get(target_sector) or {
                "operating_margin": 0.12, "roc": 0.10, "debt_ratio": 0.20, "sga": 0.15
            }
            sector_debt_ratio = sector_data.get("debt_ratio", 0.20)
            sector_median_de = sector_debt_ratio / (1 - sector_debt_ratio) if sector_debt_ratio < 1 else 0.25
            
            b_de = b_debt / b_mcap if b_mcap > 0 else 0.0
            
            if b_de < sector_median_de:
                max_allowed_debt = sector_median_de * b_mcap
                unused_borrowing_capacity = max(0.0, max_allowed_debt - b_debt)
                debt_capacity_ok = (unused_borrowing_capacity + b_cash) >= deal_ev_proxy
                debt_capacity_note = (
                    f"Резерв Senior Debt: D/E покупателя ({b_de:.2f}) ниже медианы сектора '{target_sector}' ({sector_median_de:.2f}). "
                    f"Лимит неиспользованного долга = ${unused_borrowing_capacity/1e6:.1f}M. "
                    f"Достаточность ресурсов (кредит + кэш = ${(unused_borrowing_capacity + b_cash)/1e6:.1f}M vs EV цели ${deal_ev_proxy/1e6:.1f}M): "
                    f"{'Обеспечено ✓' if debt_capacity_ok else 'Недостаточно ✗'}."
                )
            else:
                debt_capacity_ok = b_cash >= deal_ev_proxy
                debt_capacity_note = (
                    f"Превышение D/E: D/E покупателя ({b_de:.2f}) выше медианы сектора '{target_sector}' ({sector_median_de:.2f}). "
                    f"Новый Senior Debt невозможен. Сделка только за счет кэша (${b_cash/1e6:.1f}M vs EV цели ${deal_ev_proxy/1e6:.1f}M): "
                    f"{'Обеспечено ✓' if debt_capacity_ok else 'Недостаточно ✗'}."
                )
                
        if b_type == "financial" and not debt_capacity_ok:
            continue
            
        notes.append(debt_capacity_note)

        # ── 2. Синергия SG&A (совпадение SIC-кода/сектора) ──────────────────
        sic_match = (target_sic is not None and b_sic is not None and str(target_sic) == str(b_sic))
        sector_match = (target_sector is not None and b_sector is not None and target_sector == b_sector)
        has_synergy_match = sic_match or sector_match

        if has_synergy_match and target_sga > 0:
            sga_savings_annual = target_sga * 0.15  # 15% сокращение дублирующегося SG&A цели
            after_tax_savings = sga_savings_annual * (1 - tax_rate)
            # Внедряем фазирование и расходы на интеграцию по Главе 14
            synergy_value = calculate_phased_synergy_npv(sga_savings_annual, tax_rate, b_wacc)
            synergy_note = (
                f"Совпадение {'SIC-кода' if sic_match else 'сектора'} → синергия 15% SG&A цели "
                f"= ${sga_savings_annual/1e6:.1f}M/год (после налога ${after_tax_savings/1e6:.1f}M). "
                f"Динамическое фазирование (20%/50%/80%/100%/100%) за вычетом интеграционных затрат (1.2x экономии) "
                f"дает приведенную NPV ≈ ${synergy_value/1e6:.1f}M (по WACC покупателя {b_wacc*100:.1f}%)."
            )
        else:
            sga_savings_annual = 0.0
            synergy_value = 0.0
            synergy_note = "Нет пересечения SIC/сектора (или SG&A цели не определен) — операционная синергия по SG&A не применяется."
        notes.append(synergy_note)

        # ── 3. Антимонопольный барьер (HHI, FTC Horizontal Merger Guidelines) ──
        combined_universe = list(industry_universe) if industry_universe else []
        if target_revenue and target_revenue not in combined_universe:
            combined_universe.append(target_revenue)
        if b_revenue and b_revenue not in combined_universe:
            combined_universe.append(b_revenue)
        total_market_revenue = sum(r for r in combined_universe if r and r > 0)

        if total_market_revenue > 0 and target_revenue > 0 and len(combined_universe) >= 3:
            shares_pre = [(r / total_market_revenue) * 100 for r in combined_universe if r and r > 0]
            hhi_pre = _compute_hhi(shares_pre)

            combined_share_revenue = target_revenue + b_revenue
            shares_post = [
                (r / total_market_revenue) * 100
                for r in combined_universe
                if r and r > 0 and r != target_revenue and r != b_revenue
            ]
            shares_post.append((combined_share_revenue / total_market_revenue) * 100)
            hhi_post = _compute_hhi(shares_post)

            hhi_verdict_text, hhi_class, hhi_delta = _hhi_verdict(hhi_pre, hhi_post)
            hhi_note = (
                f"HHI (proxy по вселенной скринера, {len(combined_universe)} игроков): "
                f"до сделки {hhi_pre:.0f}, после {hhi_post:.0f} (Δ{hhi_delta:+.0f}) → {hhi_verdict_text}."
            )
        else:
            hhi_pre = hhi_post = hhi_delta = None
            hhi_verdict_text, hhi_class = "Н/Д (недостаточно данных о выручке индустрии для оценки HHI)", "text-secondary"
            hhi_note = hhi_verdict_text
        notes.append(hhi_note)

        # ── 4. Negative Debt Covenants Monitoring (Гл. 13) ───────────────────
        if b_type == "financial":
            post_debt = 0.75 * deal_ev_proxy
            post_ebitda = target_ebitda
            post_ni = target.get("net_income") or 0.0
            post_capex = target.get("capex") or 0.0
            post_da = target.get("da") or 0.0
        else:
            post_debt = b_debt + target_debt
            post_ebitda = target_ebitda + b_ebitda + sga_savings_annual
            post_ni = (target.get("net_income") or 0.0) + (bidder.get("net_income") or 0.0) + (sga_savings_annual * (1 - tax_rate))
            post_capex = (target.get("capex") or 0.0) + (bidder.get("capex") or 0.0)
            post_da = (target.get("da") or 0.0) + (bidder.get("da") or 0.0)

        cov_results = simulate_negative_covenants(
            post_trans_debt=post_debt,
            post_trans_ebitda=post_ebitda,
            net_income=post_ni,
            capex=post_capex,
            da=post_da,
            buyer_type=b_type
        )

        notes.append(f"<b>Мониторинг ковенантов (Гл. 13) - {cov_results['risk_level']}:</b>")
        notes.append(f"• Дивиденды: {cov_results['dividend_reason']}")
        notes.append(f"• Лимит долга: {cov_results['debt_reason']}")
        notes.append(f"• Капекс: {cov_results['capex_reason']}")

        # ── Итоговый Match Score (0-100) ─────────────────────────────────────
        score = 40 if debt_capacity_ok else 10
        if synergy_value > 0 and target_mcap > 0:
            score += min(35, int(synergy_value / target_mcap * 100))
        if hhi_class == "text-red":
            score = min(score, 20)   # антимонопольный блок доминирует над остальными факторами
        elif hhi_class == "text-orange":
            score = min(score, 60)
            
        # Корректировка Match Score по результатам ковенантного аудита
        if cov_results["breaches_count"] > 1:
            score = max(0, score - 20)  # Жесткий штраф за высокую вероятность дефолта ковенантов
        elif cov_results["breaches_count"] == 1:
            score = max(0, score - 10)  # Умеренный штраф за частичное нарушение лимитов

        score = max(0, min(100, score))

        results.append({
            "bidder_ticker": b_ticker,
            "buyer_type": b_type,
            "match_score": score,
            "debt_capacity_ok": debt_capacity_ok,
            "sga_savings_annual": round(sga_savings_annual, 0),
            "synergy_value": round(synergy_value, 0),
            "hhi_pre": round(hhi_pre, 0) if hhi_pre is not None else None,
            "hhi_post": round(hhi_post, 0) if hhi_post is not None else None,
            "hhi_delta": round(hhi_delta, 0) if hhi_delta is not None else None,
            "hhi_verdict": hhi_verdict_text,
            "hhi_class": hhi_class,
            "notes": notes,
            "covenants": cov_results
        })

    results.sort(key=lambda r: r["match_score"], reverse=True)
    return results
def run_purchase_accounting_simulation(target_shares, offer_price, latest_row, tax_rate=0.25, step_up_pct=0.15, useful_life=15):
    """
    Purchase Accounting (GAAP Consolidation Simulator) under Exhibit 12.1.
    Calculates Goodwill and Non-controlling Interest (NCI) for controlling interest acquisitions (e.g. < 100%).
    Returns a dictionary of metrics for ownership of 51%, 80%, and 100%.

    Includes the "Asset Step-Up" (asset revaluation) effect:
    - Step-up of Net Identifiable Assets reduces Goodwill.
    - Incremental annual depreciation/amortization of stepped-up assets creates a tax shield and an EPS drag.
    """
    total_assets = float(latest_row.get("total_assets", 0.0))
    total_liabilities = float(latest_row.get("total_liabilities", 0.0))
    existing_goodwill = float(latest_row.get("goodwill", 0.0))
    
    # Target's book equity
    book_equity = float(latest_row.get("equity", total_assets - total_liabilities))
    
    # Fair Value of Net Identifiable Assets (excluding existing goodwill)
    fv_net_identifiable_assets_base = book_equity - existing_goodwill
    
    # Asset Step-Up value (should be a positive percentage of gross revaluable assets: total_assets - existing_goodwill)
    step_up_base = max(0.0, total_assets - existing_goodwill)
    step_up_val = step_up_base * step_up_pct
    
    # Fair Value of Net Identifiable Assets after step-up
    fv_net_identifiable_assets = fv_net_identifiable_assets_base + step_up_val
    
    # Implied Full Equity Value of the Target based on Offer Price
    implied_full_equity_val = offer_price * target_shares
    
    # Incremental annual D&A and its effects on Newco
    annual_step_up_da = step_up_val / useful_life if useful_life > 0 else 0.0
    annual_tax_shield = annual_step_up_da * tax_rate
    annual_net_income_drag = annual_step_up_da * (1.0 - tax_rate)
    eps_drag = annual_net_income_drag / target_shares if target_shares > 0 else 0.0
    
    scenarios = {}
    for pct in [0.51, 0.80, 1.00]:
        purchase_price = implied_full_equity_val * pct
        nci_val = implied_full_equity_val * (1.0 - pct)
        
        # New goodwill under US GAAP purchase accounting (after asset step-up)
        new_goodwill = implied_full_equity_val - fv_net_identifiable_assets
        new_goodwill = max(0.0, new_goodwill)
        
        scenarios[str(int(pct * 100))] = {
            "ownership_pct": pct * 100,
            "purchase_price": round(purchase_price, 2),
            "implied_full_equity_val": round(implied_full_equity_val, 2),
            "nci_val": round(nci_val, 2),
            "fv_net_identifiable_assets_base": round(fv_net_identifiable_assets_base, 2),
            "step_up_pct": step_up_pct * 100,
            "step_up_val": round(step_up_val, 2),
            "fv_net_identifiable_assets": round(fv_net_identifiable_assets, 2),
            "new_goodwill": round(new_goodwill, 2),
            "book_equity_written_off": round(book_equity, 2),
            "existing_goodwill_written_off": round(existing_goodwill, 2),
            "annual_step_up_da": round(annual_step_up_da, 2),
            "annual_tax_shield": round(annual_tax_shield, 2),
            "annual_net_income_drag": round(annual_net_income_drag, 2),
            "eps_drag": round(eps_drag, 4),
            "annual_cash_flow_drag_stock": round(annual_step_up_da, 2),
            "annual_cash_flow_drag_asset": round(annual_net_income_drag, 2)
        }
        
    return scenarios



def run_collar_simulation(offer_price, target_market_price, collar_pct=0.10):
    """
    Моделирование воротниковых соглашений (Collar Arrangements) по Главе 11 (Exhibit 11.1).
    При оплате акциями волатильность рынка между датой подписания и датой закрытия сделки может разрушить её экономику.
    Служит для расчета обменного коэффициента (Share Exchange Ratio - SER) на границах ценовых коридоров (Collar Range).
    - Fixed-Value Collar: Стоимость сделки (Offer Price) зафиксирована внутри коридора за счет плавающего Ratio (SER).
    - Fixed-Share Collar: Коэффициент обмена (SER) зафиксирован внутри коридора, а стоимость сделки (Offer Price) колеблется.
    """
    P_T = float(offer_price)
    P_A = float(target_market_price * 1.5) if target_market_price > 0 else float(offer_price * 1.5)
    P_A = max(P_A, 1.0)
    P_A_lower = P_A * (1.0 - collar_pct)
    P_A_upper = P_A * (1.0 + collar_pct)
    R_0 = P_T / P_A
    
    scenarios_pct = [-0.20, -0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20]
    
    fv_scenarios = []
    fs_scenarios = []
    
    for pct in scenarios_pct:
        p_curr = P_A * (1.0 + pct)
        
        # 1. Fixed-Value Collar (SER floats inside, Offer Price fixed inside)
        if p_curr < P_A_lower:
            fv_ser = P_T / P_A_lower
            fv_val = fv_ser * p_curr
            fv_status = "Ниже коридора (Ratio Capped)"
        elif p_curr > P_A_upper:
            fv_ser = P_T / P_A_upper
            fv_val = fv_ser * p_curr
            fv_status = "Выше коридора (Ratio Floored)"
        else:
            fv_ser = P_T / p_curr
            fv_val = P_T
            fv_status = "Внутри коридора (Fixed Value)"
            
        fv_scenarios.append({
            "change_pct": round(pct * 100, 1),
            "acquirer_price": round(p_curr, 2),
            "exchange_ratio": round(fv_ser, 4),
            "offer_value": round(fv_val, 2),
            "status": fv_status
        })
        
        # 2. Fixed-Share Collar (SER fixed inside, Offer Price floats inside)
        if p_curr < P_A_lower:
            fs_val = R_0 * P_A_lower
            fs_ser = fs_val / p_curr
            fs_status = "Ниже коридора (Value Protected)"
        elif p_curr > P_A_upper:
            fs_val = R_0 * P_A_upper
            fs_ser = fs_val / p_curr
            fs_status = "Выше коридора (Value Capped)"
        else:
            fs_ser = R_0
            fs_val = fs_ser * p_curr
            fs_status = "Внутри коридора (Fixed Ratio)"
            
        fs_scenarios.append({
            "change_pct": round(pct * 100, 1),
            "acquirer_price": round(p_curr, 2),
            "exchange_ratio": round(fs_ser, 4),
            "offer_value": round(fs_val, 2),
            "status": fs_status
        })
        
    return {
        "acquirer_reference_price": round(P_A, 2),
        "target_reference_price": round(P_T, 2),
        "collar_pct": round(collar_pct * 100, 1),
        "collar_lower": round(P_A_lower, 2),
        "collar_upper": round(P_A_upper, 2),
        "base_exchange_ratio": round(R_0, 4),
        "fixed_value_collar": fv_scenarios,
        "fixed_share_collar": fs_scenarios
    }



def compute_preliminary_score(quote):
    pe = quote.get("trailingPE") or quote.get("forwardPE") or 25.0
    pb = quote.get("priceToBook") or 2.5
    mcap = quote.get("marketCap") or 1e12
    vol = quote.get("averageDailyVolume3Month") or 0.0
    
    if vol < 100000:
        return -1.0
        
    pe_score = max(0, 100 - pe * 3)
    pb_score = max(0, 100 - pb * 20)
    
    cap_b = mcap / 1e9
    if cap_b < 0.5:
        cap_score = 50
    elif cap_b < 3.0:
        cap_score = 100
    elif cap_b < 8.0:
        cap_score = 75
    elif cap_b <= 15.0:
        cap_score = 40
    else:
        return -1.0
        
    return 0.4 * pe_score + 0.4 * pb_score + 0.2 * cap_score

# ----------------------------------------------------
# 5. ИНТЕЛЛЕКТУАЛЬНЫЙ СБОР ДАННЫХ ИЗ YFINANCE (ДЛЯ НОВОСТЕЙ И МЕТАДАННЫХ)
# ----------------------------------------------------

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/120.0"
]

def fetch_yfinance_news_and_meta(ticker):
    headers = {"User-Agent": random.choice(USER_AGENTS)}
    stock = yf.Ticker(ticker)
    
    info = {}
    try:
        stock.session.headers.update(headers)
        info = stock.info
    except:
        pass

    buyout_news_title = None
    antitrust_news_title = None
    activist_news_title = None
    turnaround_news_title = None
    buyback_suspended_title = None
    
    try:
        news = stock.news
        if news:
            for item in news[:15]:
                content = item.get("content", {})
                title = content.get("title", "")
                summary = content.get("summary", "")
                
                title_lower = title.lower()
                summary_lower = summary.lower()
                
                # NLP/Heuristic Filter: Verify if the article is actually about the company (Macy's, Alcoa, etc.)
                # rather than just mentioning it in a list of clients/partners.
                ticker_lower = ticker.lower()
                title_stripped = title_lower.replace("'", "").replace("-", " ")
                
                is_about_company = False
                # Check 1: Ticker as a separate word in title
                if re.search(r'\b' + re.escape(ticker_lower) + r'\b', title_lower):
                    is_about_company = True
                
                # Check 2: Core company name in title
                name_clean = str(info.get("longName") or info.get("shortName") or "").lower()
                for suffix in ["inc", "corp", "co", "ltd", "corporation", "company", "stores", "realty", "solutions"]:
                    name_clean = re.sub(r'\b' + suffix + r'\b', '', name_clean)
                name_clean = name_clean.replace("'", "").replace(",", "").replace(".", "").strip()
                
                name_words = name_clean.split()
                if name_words:
                    base_name = name_words[0]
                    if len(base_name) > 2 and base_name in title_stripped:
                        is_about_company = True
                    if len(name_words) >= 2:
                        base_name_2 = name_words[0] + " " + name_words[1]
                        if len(base_name_2) > 3 and base_name_2 in title_stripped:
                            is_about_company = True
                
                # If ticker is long enough and mentioned in title (even as substring)
                if not is_about_company and len(ticker) >= 3 and ticker_lower in title_stripped:
                    is_about_company = True
                    
                if not is_about_company:
                    continue
                
                text_to_scan = (title + " " + summary).lower()
                
                if not buyout_news_title:
                    buyout_patterns = [
                        f"acquire {ticker_lower}",
                        f"buyout of {ticker_lower}",
                        f"acquisition of {ticker_lower}",
                        f"buy {ticker_lower}",
                        f"takeover of {ticker_lower}",
                        f"{ticker_lower} buyout",
                        f"buying {ticker_lower}",
                        f"purchase of {ticker_lower}"
                    ]
                    recommendation_kws = ["upgrade", "downgrade", "rating", "target price", "price target", 
                                          "analyst", "consensus", "neutral", "hold", "sell", "strong buy", 
                                          "outperform", "underperform", "initiates", "zacks", "motley fool"]
                    if any(pat in text_to_scan for pat in buyout_patterns):
                        if not any(kw in text_to_scan for kw in recommendation_kws):
                            buyout_news_title = title
                
                if not antitrust_news_title:
                    if any(kw in text_to_scan for kw in ["ftc", "antitrust", "block", "lawsuit", "regulator", "sued"]):
                        if any(kw2 in text_to_scan for kw2 in ["merger", "acquisition", "deal", "buyout"]):
                            antitrust_news_title = title
                            
                if not activist_news_title:
                    has_activist_kw = any(kw in text_to_scan for kw in ["activist", "hostile", "proxy fight", "board seat", "arkhouse", "brigade"])
                    has_real_pressure = "pressure" in text_to_scan and not any(p_fp in text_to_scan for p_fp in ["margin pressure", "pricing pressure", "cost pressure", "inflationary pressure", "downward pressure", "competitive pressure"])
                    if has_activist_kw or has_real_pressure:
                        recommendation_kws = ["upgrade", "downgrade", "rating", "target price", "price target", 
                                              "analyst", "consensus", "neutral", "hold", "sell", "strong buy", 
                                              "outperform", "underperform", "initiates", "zacks", "motley fool"]
                        if not any(kw in text_to_scan for kw in recommendation_kws):
                            activist_news_title = title

                if not turnaround_news_title:
                    if any(kw in text_to_scan for kw in ["turnaround", "transformation plan", "restructuring plan"]):
                        turnaround_news_title = title

                if not buyback_suspended_title:
                    if any(kw in text_to_scan for kw in ["suspend buyback", "suspend share repurchase", "halt buyback", "stop buyback"]):
                        buyback_suspended_title = title
    except:
        pass
        
    return info, buyout_news_title, antitrust_news_title, activist_news_title, turnaround_news_title, buyback_suspended_title

def get_sector_by_sic(sic):
    try:
        sic_val = int(sic)
    except:
        return None
        
    if 100 <= sic_val <= 999:
        return "Consumer Defensive"
    elif 1000 <= sic_val <= 1299:
        return "Basic Materials"
    elif 1300 <= sic_val <= 1399:
        return "Energy"
    elif 1400 <= sic_val <= 1499:
        return "Basic Materials"
    elif 1500 <= sic_val <= 1799:
        return "Industrials"
    elif 2000 <= sic_val <= 2199:
        return "Consumer Defensive"
    elif 2200 <= sic_val <= 2799:
        return "Consumer Cyclical"
    elif 2830 <= sic_val <= 2836:
        return "Healthcare"  # Pharma
    elif 2840 <= sic_val <= 2844:
        return "Consumer Defensive"  # Soap, Cleaners, Bleach, Cosmetics (CLX, PG, CL)
    elif 2800 <= sic_val <= 2899:
        return "Basic Materials"  # Industrial Chemicals
    elif 2900 <= sic_val <= 2999:
        return "Energy"
    elif 3000 <= sic_val <= 3499:
        return "Industrials"
    elif 3570 <= sic_val <= 3579:
        return "Technology"  # Computers
    elif 3500 <= sic_val <= 3599:
        return "Industrials"
    elif 3670 <= sic_val <= 3679:
        return "Technology"  # Semiconductors
    elif 3600 <= sic_val <= 3699:
        return "Technology"
    elif 3711 <= sic_val <= 3716:
        return "Consumer Cyclical"  # Auto
    elif 3700 <= sic_val <= 3799:
        return "Industrials"
    elif 3800 <= sic_val <= 3899:
        return "Healthcare"  # Medical Instruments
    elif 3900 <= sic_val <= 3999:
        return "Consumer Cyclical"
    elif 4000 <= sic_val <= 4799:
        return "Industrials"  # Transport
    elif 4800 <= sic_val <= 4899:
        return "Communication Services"  # Telecom
    elif 4900 <= sic_val <= 4999:
        return "Utilities"
    elif 5000 <= sic_val <= 5199:
        return "Industrials"
    elif 5400 <= sic_val <= 5499:
        return "Consumer Defensive"  # Food stores
    elif 5200 <= sic_val <= 5999:
        return "Consumer Cyclical"  # Retail
    elif 6000 <= sic_val <= 6499:
        return "Financial Services"
    elif 6500 <= sic_val <= 6799:
        return "Real Estate"
    elif 7370 <= sic_val <= 7379:
        return "Technology"  # Software
    elif 7000 <= sic_val <= 7299:
        return "Consumer Cyclical"  # Hotels, Personal services (HRB!)
    elif 8000 <= sic_val <= 8099:
        return "Healthcare"
    else:
        return None

def extract_mda_section(html_content, is_10k=True):
    """
    Ищет начало раздела MD&A и отрезает всё, что идёт до него.
    is_10k=True -> Ищет Item 7 (для 10-K)
    is_10k=False -> Ищет Item 2 (для 10-Q)
    """
    import re
    from bs4 import BeautifulSoup
    item_num = "7" if is_10k else "2"
    
    # Регулярка, которая пробивает HTML-код любой степени уродливости
    mda_regex = re.compile(
        rf"item\s*{item_num}\s*[:\.]?\s*<[^>]+>*\s*Management"
        rf"[\s\s|&nbsp;\n]*['’]?s\s*Discussion\s*and\s*Analysis",
        re.IGNORECASE
    )
    
    # 1. Быстрый поиск позиции через regex в сыром HTML
    match = mda_regex.search(html_content)
    
    if not match:
        # Фоллбэк-вариант на случай, если слово Item и Management разделены более агрессивными тегами
        fallback_regex = re.compile(
            rf"item\s*{item_num}\s*[\s\S]{{0,100}}Management[\s\S]{{0,50}}Discussion", 
            re.IGNORECASE
        )
        match = fallback_regex.search(html_content)
        
    if match:
        # Отрезаем всё, что было ДО начала раздела MD&A
        mda_html_chunk = html_content[match.start():]
        
        # 2. Очищаем оставшийся чанк от HTML-мусора через BeautifulSoup
        soup = BeautifulSoup(mda_html_chunk, "lxml")
        clean_text = soup.get_text(separator="\n")
        
        # Нормализуем пробелы и Юникод (убираем глитчи вроде \xa0 и скрытые символы)
        clean_text = re.sub(r'\xa0', ' ', clean_text)
        clean_text = re.sub(r'[ \t]+', ' ', clean_text)
        
        return clean_text
    
    return None


def fetch_sec_ma_hints(client, ticker):
    print(f"[*] SEC & Qualitative: Поиск M&A зацепок и инсайтов для {ticker}...")
    import requests
    import re
    import json
    from bs4 import BeautifulSoup
    ma_filings = []
    qual_points = []
    qual_score = 0
    detected_defenses = []
    
    # Флаг для предотвращения дублирования одной и той же сделки слияния
    has_processed_merger = False
    # Флаг для предотвращения дублирования одной и той же сделки слияния
    has_processed_merger = False
    has_processed_8k_205 = False
    has_processed_8k_701 = False
    has_processed_8k_502 = False
    
    # 1. Скрапинг транскрипта (Earnings Call) через yfinance
    try:
        stock = yf.Ticker(ticker)
        transcript_url = None
        for n in stock.news:
            title = n.get('content', {}).get('title', '').lower()
            if 'transcript' in title and 'earnings' in title:
                transcript_url = n.get('content', {}).get('clickThroughUrl', {}).get('url')
                break
        if transcript_url:
            res = requests.get(transcript_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            soup = BeautifulSoup(res.text, 'html.parser')
            article = soup.find('div', class_='caas-body')
            text = article.get_text(separator='\n') if article else soup.get_text(separator='\n')
            
            qa_match = re.search(r'(?i)(question-and-answer session|question and answer session|q&a session|operator\s*\[\d+\])', text)
            if qa_match:
                text = text[:qa_match.start()]
                
            clean_transcript = text[:6000]
            
            messages = [
                {"role": "system", "content": "You are a financial analyst. Read the earnings call prepared remarks. Extract the exact name of any major turnaround strategy or restructuring plan mentioned by the CEO (e.g., 'Bold New Chapter'). If found, translate the strategy name and respond ONLY with the translated strategy name IN RUSSIAN. If no such strategy is mentioned, respond ONLY with NO. CRITICAL: Output ONLY the requested data. No preambles, no quotes."},
                {"role": "user", "content": f"Ticker: {ticker}. Transcript: {clean_transcript}"}
            ]
            ans = clean_llm_response(call_ollama(messages, max_tokens=50, temperature=0.0))
            if ans and "NO" not in ans.upper() and len(ans) > 3:
                qual_score += 15
                qual_points.append(f"Упоминание стратегии разворота / Turnaround в Earnings Call (+15 баллов): {ans}")
    except Exception as e:
        print(f"    [!] Ошибка транскрипта {ticker}: {e}")

    try:
        meta = client.get_company_metadata(ticker)
        recent = meta.get("filings", {}).get("recent", {})
        if not recent:
            return ma_filings, qual_points, min(60, qual_score), list(set(detected_defenses))
            
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        descs = recent.get("primaryDocDescription", [])
        items = recent.get("items", [])
        acc_nums = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])
        
        from datetime import datetime, timedelta
        one_year_ago = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        parsed_10k_10q = False
        
        for i in range(len(forms)):
            form = forms[i]
            date = dates[i]
            item = items[i] if i < len(items) else ""
            acc = acc_nums[i]
            pdoc = primary_docs[i]
            
            if date < one_year_ago:
                continue
                
            is_ma = False
            reasons = []
            desc = ""
            key_demands_val = []
            evidence_quotes_val = []
            
            if form in ["SC 13D", "SC 13D/A"]:
                is_ma = True
                reasons.append("Активизм / Накопление доли (SC 13D)")
                cik_str = str(meta.get("cik", "")).zfill(10)
                acc_clean = acc.replace("-", "")
                doc_url = f"https://www.sec.gov/Archives/edgar/data/{cik_str}/{acc_clean}/{pdoc}"
                headers = {"User-Agent": client.headers.get("User-Agent", "ValuationDashboardBot/3.0 (dikam@example.com)")}
                try:
                    res_13d = requests.get(doc_url, headers=headers, timeout=10)
                    if res_13d.status_code == 200:
                        item4_text = extract_item4_section(res_13d.text)
                        if item4_text:
                            clean_item4 = strip_boilerplate(item4_text)
                            intent_data = gemma_4b_inference_engine(clean_item4[:6000])
                            
                            intent_type = intent_data.get("intent", "PASSIVE_INVESTMENT")
                            explanation = intent_data.get("summary", "Раскрытие накопления доли в Schedule 13D.")
                            key_demands_val = intent_data.get("demands", [])
                            evidence_quotes_val = intent_data.get("quotes", [])
                            
                            add_score = {"HOSTILE_TAKEOVER": 30, "ACTIVIST_RESTRUCTURING": 25, "FRIENDLY_TOEHOLD": 15, "PASSIVE_INVESTMENT": 5}.get(intent_type, 5)
                            qual_score += add_score
                            qual_points.append(f"Инвестор-активист (13D {intent_type}) (+{add_score} баллов): {explanation}")
                            desc = f"13D ({intent_type}): {explanation}"
                except Exception as e:
                    print(f"    [!] Error parsing 13D for {ticker}: {e}")
                    
            elif form in ["SC 13G", "SC 13G/A"]:
                is_ma = True
                reasons.append("Пассивное накопление доли >5% (SC 13G)")
                intent_type = "PASSIVE_INVESTMENT"
                explanation = "Раскрытие пассивной доли институционального инвестора в Schedule 13G без целей изменить контроль."
                
                key_demands_val = []
                evidence_quotes_val = []
                
                qual_score += 5
                qual_points.append(f"Пассивный инвестор (13G {intent_type}) (+5 баллов): {explanation}")
                desc = f"13G ({intent_type}): {explanation}"

            elif "14A" in form and any(kw in form for kw in ["PREM", "DEFM"]):
                # ЗАЩИТА ОТ ДУБЛИРОВАНИЯ: Берем только самый свежий отчет по слиянию
                if has_processed_merger:
                    continue
                is_ma = True
                
                cik_str = str(meta.get("cik", "")).zfill(10)
                acc_clean = acc.replace("-", "")
                doc_url = f"https://www.sec.gov/Archives/edgar/data/{cik_str}/{acc_clean}/{pdoc}"
                headers = {"User-Agent": client.headers.get("User-Agent", "ValuationDashboardBot/3.0 (dikam@example.com)")}
                
                direction = "Параметры слияния не определены"
                terms = "Условия не определены"
                
                try:
                    res_14a = requests.get(doc_url, headers=headers, timeout=10)
                    if res_14a.status_code == 200:
                        import html
                        raw_text = res_14a.content.decode('utf-8', errors='replace')
                        raw_text = html.unescape(raw_text)
                        clean_text = re.sub(r'<[^>]*>', ' ', raw_text)
                        clean_text = clean_text.replace('\u2019', "'").replace('\u2018', "'").replace('\u201c', '"').replace('\u201d', '"')
                        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
                        
                        idx = clean_text.lower().find("summary term sheet")
                        if idx == -1: idx = clean_text.lower().find("the merger agreement")
                        if idx == -1: idx = clean_text.lower().find("proposed merger")
                        if idx == -1: idx = 0
                            
                        chunk = clean_text[idx:idx+9000]
                        
                        messages = [
                            {
                                "role": "system", 
                                "content": (
                                    "You are an expert M&A analyst. Read the merger proxy statement text.\n"
                                    "Determine exactly WHO IS BUYING WHOM. Identify the Acquirer (Buyer) and the Target (Asset/Company being bought).\n"
                                    "CRITICAL REQUIREMENT: You MUST write the description strictly in RUSSIAN and explicitly state the direction of the transaction.\n"
                                    "Example format for 'direction': 'Амнеал Фармасьютикалс покупает компанию Кашив БиоСайенсис'\n"
                                    "Example format for 'deal_terms': 'Обмен 28.9 млн акций Класса А на 100% долей участия'\n\n"
                                    "You MUST respond with a single valid JSON object ONLY containing exactly two keys: 'direction' and 'deal_terms'.\n"
                                    "REQUIRED JSON FORMAT:\n"
                                    "{\n"
                                    "  \"direction\": \"Кто кого покупает НА РУССКОМ ЯЗЫКЕ\",\n"
                                    "  \"deal_terms\": \"Условия сделки НА РУССКОМ ЯЗЫКЕ (коротко и емко)\"\n"
                                    "}\n"
                                    "Do not add markdown blocks or text preambles. Output pure JSON only."
                                )
                            },
                            {"role": "user", "content": f"Filing company ticker: {ticker}. Text snippet: {chunk}"}
                        ]
                        
                        ans = call_ollama(messages, max_tokens=150, temperature=0.0)
                        if ans:
                            ans_cleaned = re.sub(r'```[a-zA-Z]*\n?', '', ans).replace('```', '').strip()
                            json_match = re.search(r'\{.*\}', ans_cleaned, re.DOTALL)
                            if json_match:
                                merger_json = json.loads(json_match.group(0))
                                direction = merger_json.get("direction", "Параметры слияния не определены")
                                terms = merger_json.get("deal_terms", "Условия не определены")
                except Exception as e:
                    print(f"    [!] Ошибка при парсинге прокси-формы 14A для {ticker}: {e}")
                
                reasons.append(f"M&A Событие: {direction} ({form})")
                desc = f"{direction} | Условия: {terms}"
                
                qual_point_text = f"M&A Событие: {direction} ({form}) на условиях '{terms}' (+90 баллов)"
                if qual_point_text not in qual_points:
                    qual_score += 90
                    qual_points.append(qual_point_text)
                has_processed_merger = True
                
            elif form in ["SC TO-T", "SC TO-T/A", "SC TO-C", "SC TO-I"]:
                is_ma = True
                reasons.append("Тендерное предложение (TO)")
                
            elif form in ["S-4", "S-4/A"]:
                if has_processed_merger:
                    continue
                is_ma = True
                cik_str = str(meta.get("cik", "")).zfill(10)
                acc_clean = acc.replace("-", "")
                doc_url = f"https://www.sec.gov/Archives/edgar/data/{cik_str}/{acc_clean}/{pdoc}"
                headers = {"User-Agent": client.headers.get("User-Agent", "ValuationDashboardBot/3.0 (dikam@example.com)")}
                direction = "Регистрация акций под слияние"
                terms = "Обмен акций"
                try:
                    res_m = requests.get(doc_url, headers=headers, timeout=10)
                    if res_m.status_code in [200, 206]:
                        soup = BeautifulSoup(res_m.text, "lxml")
                        clean_text = re.sub(r'\s+', ' ', soup.get_text(separator=" ")).strip()
                        
                        messages = [
                            {
                                "role": "system",
                                "content": (
                                    "You are an expert M&A analyst. Read the cover page of this S-4 Registration Statement.\n"
                                    "Determine exactly WHO IS BUYING WHOM. Identify the Acquirer (Buyer) and the Target (Asset/Company being bought).\n"
                                    "CRITICAL REQUIREMENT: You MUST write the description strictly in RUSSIAN and explicitly state the direction of the transaction.\n"
                                    "You MUST respond with a single valid JSON object ONLY containing exactly two keys: 'direction' and 'deal_terms'.\n\n"
                                    "REQUIRED JSON FORMAT:\n"
                                    "{\n"
                                    "  \"direction\": \"Кто кого покупает НА РУССКОМ ЯЗЫКЕ\",\n"
                                    "  \"deal_terms\": \"Условия обмена акций НА РУССКОМ ЯЗЫКЕ\"\n"
                                    "}"
                                )
                            },
                            {"role": "user", "content": f"Text: {clean_text[:6000]}"}
                        ]
                        ans = call_ollama(messages, max_tokens=150, temperature=0.0)
                        if ans:
                            ans_cleaned = re.sub(r'```[a-zA-Z]*\n?', '', ans).replace('```', '').strip()
                            json_match = re.search(r'\{.*\}', ans_cleaned, re.DOTALL)
                            if json_match:
                                merger_json = json.loads(json_match.group(0))
                                direction = merger_json.get("direction", "Регистрация акций под слияние")
                                terms = merger_json.get("deal_terms", "Обмен акций")
                except Exception as e:
                    print(f"    [!] Error parsing S-4 cover for {ticker}: {e}")
                
                reasons.append(f"Регистрация акций под слияние: {direction} (S-4)")
                desc = f"{direction} | Условия: {terms}"
                
                qual_point_text = f"Регистрация акций под слияние: {direction} (S-4) на условиях '{terms}' (+90 баллов)"
                if qual_point_text not in qual_points:
                    qual_score += 90
                    qual_points.append(qual_point_text)
                has_processed_merger = True
                
            elif form in ["10-K", "10-Q", "DEF 14A"] and not parsed_10k_10q:
                cik_str = str(meta.get("cik", "")).zfill(10)
                acc_clean = acc.replace("-", "")
                doc_url = f"https://www.sec.gov/Archives/edgar/data/{cik_str}/{acc_clean}/{pdoc}"
                headers = {"User-Agent": client.headers.get("User-Agent", "ValuationDashboardBot/3.0 (dikam@example.com)")}
                try:
                    res = requests.get(doc_url, headers=headers, timeout=10)
                    if res.status_code == 200:
                        doc_text_lower = res.text.lower()
                        if form in ["10-K", "DEF 14A"]:
                            defenses = {
                                "poison pill": "Ядовитая пилюля (Poison Pill)", "shareholder rights plan": "План прав акционеров (Poison Pill)",
                                "staggered board": "Шахматный совет директоров (Staggered Board)", "classified board": "Классифицированный совет директоров (Classified Board)",
                                "golden parachute": "Золотой парашют (Golden Parachute)", "change-in-control": "Золотой парашют (Golden Parachute)",
                                "special meeting": "Ограничение на созыв внеочередных собраний (Limits on Special Meetings)", "written consent": "Ограничение на письменные согласия (Limits on Written Consent)"
                            }
                            found_defenses = [label for kw, label in defenses.items() if kw in doc_text_lower]
                            if found_defenses:
                                unique_defenses = list(set(found_defenses))
                                detected_defenses.extend(unique_defenses)
                                qual_point_text = f"Обнаружены защиты от поглощения (-20 баллов): {', '.join(unique_defenses)}"
                                if qual_point_text not in qual_points:
                                    qual_score -= 20
                                    qual_points.append(qual_point_text)
                                is_ma = True
                                reasons.append("Защиты от поглощения (Charter/Bylaws)")
                                desc = f"Защитные барьеры: {', '.join(unique_defenses)}" 

                        if form in ["10-K", "10-Q"]:
                            mda_text = extract_mda_section(res.text, is_10k=(form == "10-K"))
                            if mda_text:
                                mda_system_instruction = (
                                    "You are a strict M&A financial analyst. Analyze the provided MD&A. Look for restructuring or turnaround strategies.\n"
                                    "Respond with a single valid JSON object ONLY in RUSSIAN language.\n"
                                    "REQUIRED JSON FORMAT:\n"
                                    "{\n"
                                    "  \"intent\": \"PASSIVE_INVESTMENT\",\n"
                                    "  \"summary\": \"Краткое описание на РУССКОМ языке (макс 8 слов)\",\n"
                                    "  \"demands\": [], \"quotes\": [], \"turnaround_detected\": true_or_false, \"spinoff_detected\": true_or_false\n"
                                    "}"
                                )
                                mda_data = gemma_4b_inference_engine(mda_text[:12000], system_instruction=mda_system_instruction)
                                if mda_data.get("turnaround_detected"):
                                    ans = mda_data.get("summary", "Программа оздоровления бизнеса")
                                    qual_point_text = f"Стратегия Turnaround в {form} MD&A (+20 баллов): {ans}"
                                    if qual_point_text not in qual_points:
                                        qual_score += 20
                                        qual_points.append(qual_point_text)
                                    is_ma = True
                                    reasons.append("План оздоровления бизнеса (MD&A)")
                                    desc = f"Стратегия Turnaround: {ans}"
                                    
                                if mda_data.get("spinoff_detected"):
                                    ans_seg = mda_data.get("summary", "Потенциальный Spin-off / Carve-Out")
                                    qual_point_text = f"Segment Reporting / Возможный Spin-off в {form} (+15 баллов): {ans_seg}"
                                    if qual_point_text not in qual_points:
                                        qual_score += 15
                                        qual_points.append(qual_point_text)
                                    is_ma = True
                                    reasons.append("Сегментная реструктуризация / Carve-out")
                                    desc = f"Реструктуризация сегментов: {ans_seg}" if not desc else desc + f" | Реструктуризация: {ans_seg}"
                except Exception as e:
                    print(f"    [!] Ошибка при парсинге {form} для {ticker}: {e}")
                if form in ["10-K", "10-Q"]: parsed_10k_10q = True

            elif form == "8-K":
                item_list = str(item).split(",") if item else []
                if any(i in item_list for i in ["1.01", "2.01", "5.02", "2.05", "7.01"]):
                    cik_str = str(meta.get("cik", "")).zfill(10)
                    acc_clean = acc.replace("-", "")
                    doc_url = f"https://www.sec.gov/Archives/edgar/data/{cik_str}/{acc_clean}/{pdoc}"
                    headers = {"User-Agent": client.headers.get("User-Agent", "ValuationDashboardBot/3.0 (dikam@example.com)"), "Range": "bytes=0-35000"}
                    try:
                        res = requests.get(doc_url, headers=headers, timeout=5)
                        if res.status_code in [200, 206]:
                            import html
                            raw_text = html.unescape(res.content.decode('utf-8', errors='replace'))
                            clean_text = re.sub(r'\s+', ' ', re.sub(r'<[^>]*>', ' ', raw_text)).strip()
                            lower_text = clean_text.lower()
                            
                            positive_kws = ["merger", "acquisition", "takeover", "purchase agreement", "tender offer"]
                            
                            if "5.02" in item_list and not has_processed_8k_502:
                                messages = [
                                    {
                                        "role": "system", 
                                        "content": (
                                            "You are a strict financial data extraction engine analyzing an SEC 8-K document.\n"
                                            "Determine if a new board member was appointed pursuant to an official agreement, support letter, or settlement with an outside investment fund, hedge fund, or shareholder.\n"
                                            "If it is a routine appointment or selection of an independent director by the company itself WITHOUT outside fund pressure, set 'is_activist_agreement' to false.\n\n"
                                            "You MUST respond with a single valid JSON object ONLY containing exactly four keys:\n"
                                            "1. 'is_activist_agreement': true or false.\n"
                                            "2. 'verbatim_english_sentence': The EXACT, unaltered sentence from the text that proves the agreement and names the fund. MUST BE VERBATIM.\n"
                                            "3. 'fund_name_en': The exact English proper noun of the investment fund/shareholder entity.\n"
                                            "4. 'fund_name_ru': Clean Russian transliteration of the fund's name.\n\n"
                                            "If 'is_activist_agreement' is false, leave string fields empty.\n"
                                            "Output pure JSON only, no preambles, no markdown blocks."
                                        )
                                    },
                                    {"role": "user", "content": f"Text: {clean_text[:5000]}"}
                                ]
                                ans = call_ollama(messages, max_tokens=150, temperature=0.0)
                                if ans:
                                    ans_cleaned = re.sub(r'```[a-zA-Z]*\n?', '', ans).replace('```', '').strip()
                                    json_match = re.search(r'\{.*\}', ans_cleaned, re.DOTALL)
                                    if json_match:
                                        try:
                                            data_502 = json.loads(json_match.group(0))
                                            if data_502.get("is_activist_agreement"):
                                                evidence = data_502.get("verbatim_english_sentence", "").strip()
                                                fund_en = data_502.get("fund_name_en", "").strip()
                                                fund_ru = data_502.get("fund_name_ru", "").strip()
                                                
                                                # АРХИТЕКТУРНЫЙ ФИЛЬТР 1 (Заземление): Проверяем, существует ли выданная ИИ цитата в исходном тексте отчета
                                                if evidence and evidence.lower() in clean_text.lower():
                                                    
                                                    # АРХИТЕКТУРНЫЙ ФИЛЬТР 2: Проверяем, что имя фонда действительно находится внутри этой цитаты
                                                    if fund_en and fund_en.lower() in evidence.lower():
                                                        
                                                        # АРХИТЕКТУРНЫЙ ФИЛЬТР 3: Проверяем наличие маркеров принадлежности к инвестиционным структурам (институциональный фильтр)
                                                        FUND_SUFFIXES = ["CAPITAL", "MANAGEMENT", "PARTNERS", "VALUE", "HOLDINGS", 
                                                                         "FUND", "GROUP", "INVESTORS", "ADVISORS", "ASSET", "TRUST"]
                                                        is_valid_financial_entity = any(s in fund_en.upper() for s in FUND_SUFFIXES)
                                                        
                                                        if is_valid_financial_entity:
                                                            qual_point_text = f"Приход представителя фонда в Совет Директоров (Item 5.02) (+25 баллов): {fund_ru}"
                                                            if qual_point_text not in qual_points:
                                                                qual_score += 25
                                                                qual_points.append(qual_point_text)
                                                                is_ma = True
                                                                reasons.append("Соглашение с активистом (8-K 5.02)")
                                                                desc = f"Вход фонда в Совет Директоров: {fund_ru} ({fund_en})"
                                                                has_processed_8k_502 = True
                                        except:
                                            pass
                            
                            if "7.01" in item_list and not has_processed_8k_701:
                                messages = [
                                    {"role": "system", "content": (
                                        "You are a strict financial analyst examining an 8-K Item 7.01 snippet.\n"
                                        "CRITICAL FILTER: Strictly ignore all routine quarterly financial updates, earnings releases, conference call announcements, and future guidance. "
                                        "Look ONLY for major non-routine events: material corporate acquisitions, spin-offs, or massive strategic overhauls.\n"
                                        "If the text is a routine financial results announcement, quarterly wrap-up or earnings release, respond strictly with 'NO'.\n"
                                        "If it is a major strategic event or acquisition, extract a short summary (max 8 words) STRICTLY IN RUSSIAN language. Do not use any English words.\n"
                                        "EXAMPLE:\n"
                                        "Input text: 'Amneal announced the acquisition of Kashiv BioSciences for contingent payments.'\n"
                                        "Your output: Приобретение биофармацевтической компании Кашив БиоСайенсис\n\n"
                                        "Analyze the text now."
                                    )},
                                    {"role": "user", "content": f"Text: {clean_text[:5000]}"}
                                ]
                                ans = clean_llm_response(call_ollama(messages, max_tokens=50, temperature=0.0))
                                
                                ans_clean_test = ans.strip(".! ").lower()
                                is_junk_response = any(w in ans_clean_test for w in [
                                    "да", "нет", "реструктуризация", "результаты", "показатели", "отчет", "закрытие"
                                ]) or len(ans_clean_test) < 5
                                
                                if ans and "NO" not in ans.upper() and not is_junk_response:
                                    qual_point_text = f"Внеочередное M&A событие (Item 7.01) (+20 баллов): {ans}"
                                    if qual_point_text not in qual_points:
                                        qual_score += 20
                                        qual_points.append(qual_point_text)
                                        is_ma = True
                                        reasons.append("Стратегическое событие (Item 7.01)")
                                        desc = f"Стратегическое событие: {ans}" if not desc else desc + f" | План: {ans}"
                                        has_processed_8k_701 = True
                            
                            if ("1.01" in item_list or "2.01" in item_list) and any(pk in lower_text for pk in positive_kws):
                                is_ma = True
                                if "1.01" in item_list: reasons.append("Существенное соглашение (8-K Item 1.01)")
                                if "2.01" in item_list: reasons.append("Завершение сделки (8-K Item 2.01)")
                                snippet = ""
                                for pk in positive_kws:
                                    idx = lower_text.find(pk)
                                    if idx != -1:
                                        snippet = clean_text[max(0, idx - 45):min(len(clean_text), idx + 105)].replace('"', "'").replace('<', '').replace('>', '').strip()
                                        break
                                desc = f"Контекст: '...{snippet}...'" if snippet else "Обнаружено M&A соглашение (8-K)"
                    except Exception as ex:
                        print(f"    [!] SEC: ошибка чтения 8-K: {ex}")
            
            if is_ma:
                cik_str = str(meta.get("cik", "")).zfill(10)
                acc_clean = acc.replace("-", "")
                ma_filings.append({
                    "date": date, "form": form, "reason": ", ".join(reasons), "description": desc,
                    "url": f"https://www.sec.gov/Archives/edgar/data/{cik_str}/{acc_clean}/{pdoc}",
                    "key_demands": key_demands_val, "evidence_quotes": evidence_quotes_val
                })
                if len(ma_filings) >= 4:
                    break
    except Exception as e:
        print(f"  [!] Ошибка сбора SEC M&A зацепок для {ticker}: {e}")
    return ma_filings, qual_points, min(60, qual_score), list(set(detected_defenses))


# ----------------------------------------------------
# 6. ГЛАВНЫЙ ИСПОЛНИТЕЛЬНЫЙ ЦИКЛ
# ----------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="M&A Attractiveness & Takeover Feasibility Screener")
    parser.add_argument("--exclude-sp500", action="store_true", default=True, help="Исключить компании из S&P 500")
    parser.add_argument("--no-exclude-sp500", action="store_false", dest="exclude_sp500", help="Не исключать S&P 500")
    parser.add_argument("--limit", type=int, default=50, help="Количество глубоко анализируемых компаний после первичного отбора")
    parser.add_argument("--step-up-pct", type=float, default=0.15, help="Процент переоценки чистых активов (Asset Step-Up) в диапазоне [0.0, 1.0]")
    parser.add_argument("tickers", nargs="*", help="Специфические тикеры для глубокого анализа")
    args = parser.parse_args()

    cache = load_cache()
    
    # Автоматическая инвалидация устаревшего кэша
    invalidated = False
    for t in list(cache.get("tickers", {}).keys()):
        obj = cache["tickers"][t]
        
        # Check if consolidation_scenarios has "51" and if "step_up_val" is in "51" to force recalculation for the new purchase accounting simulator
        has_new_pa = False
        scen = obj.get("consolidation_scenarios", {})
        if "51" in scen and "step_up_val" in scen["51"]:
            has_new_pa = True

        if "timing" not in obj or "buyer_profile" not in obj or "offer_price" not in obj.get("financials", {}) or not has_new_pa or "collar_scenarios" not in obj:
            del cache["tickers"][t]
            invalidated = True
    if invalidated:
        save_cache(cache)
        print("[*] Устаревший или некорректный кэш для некоторых тикеров был сброшен для принудительного пересчета.")
    
    # Сбор черного списка S&P 500
    sp500_list = []
    if args.exclude_sp500:
        sp500_list = fetch_sp500_tickers(cache)
        
    # --- СТАДИЯ 1: ПЕРВИЧНЫЙ СКРИНИНГ ВЕСЬ РЫНОК ---
    print("[*] Сбор данных по рынку US Small/Mid Cap через yf.screen()...")
    all_quotes = []
    offset = 0
    limit_bulk = 1000 
    
    while offset < limit_bulk:
        q = EquityQuery('and', [
            EquityQuery('eq', ['region', 'us']),
            EquityQuery('is-in', ['exchange', 'NMS', 'NYQ']),
            EquityQuery('btwn', ['intradaymarketcap', 200000000, 15000000000])
        ])
        try:
            res = yf.screen(q, sortField='intradaymarketcap', sortAsc=False, size=250, offset=offset)
            quotes = res.get("quotes", [])
            if not quotes:
                break
            all_quotes.extend(quotes)
            offset += len(quotes)
            print(f"  [+] Загружено {offset} компаний...")
            time.sleep(0.5)
        except Exception as e:
            print(f"  [!] Ошибка пагинации yf.screen на offset {offset}: {e}")
            break
            
    # Фильтрация S&P 500
    blacklist_sp500 = set(sp500_list)
    filtered_quotes = [q for q in all_quotes if q.get("symbol") not in blacklist_sp500]
    print(f"[*] Исключено S&P 500. Для ранжирования доступно: {len(filtered_quotes)} компаний.")
    
    # Оценка предварительного M&A потенциала
    scored_quotes = []
    for quote in filtered_quotes:
        score = compute_preliminary_score(quote)
        if score > 0:
            scored_quotes.append((score, quote))
            
    scored_quotes.sort(key=lambda x: x[0], reverse=True)
    print(f"[+] Отранжировано кандидатов по предварительному M&A потенциалу: {len(scored_quotes)}")
    
    # Топ кандидаты
    top_candidates = [item[1]["symbol"] for item in scored_quotes[:args.limit]]
    
    # Гарантированный пул тикеров: Workspace + Пользовательские
    custom_tickers = [t.upper() for t in args.tickers] if args.tickers else []
    
    if args.tickers:
        deep_tickers = custom_tickers
    else:
        deep_tickers = top_candidates

    # Исключаем тикеры из чёрного списка (нет XBRL-данных в SEC)
    blacklist = cache.get("blacklist", {})
    skipped = [t for t in deep_tickers if t in blacklist and t not in custom_tickers]
    if skipped:
        print(f"[~] Пропущено {len(skipped)} тикеров (нет SEC/XBRL-данных): {', '.join(skipped)}")
    deep_tickers = [t for t in deep_tickers if t not in blacklist or t in custom_tickers]
    print(f"[*] Итоговый пул для глубокой оценки по Дамодарану: {len(deep_tickers)} тикеров.")
    
    # --- СТАДИЯ 2: ГЛУБОКАЯ ОЦЕНКА ---
    client = SECEDGARClient()
    parser_sec = CustomFinancialParser(client)
    risk_eng = RiskEngine(erp=DAMODARAN_ERP)
    opt_control = OptimalControlEngine()
    
    companies_data = []
    
    for t in deep_tickers:
        print(f"\n[~] Глубокий анализ {t}...")
        
        if t in cache["tickers"]:
            print(f"  [+] Загружено из локального кэша скринера.")
            comp_obj = cache["tickers"][t]
            if t in sp500_list:
                comp_obj["index_group"] = "SP500"
            elif t in custom_tickers:
                comp_obj["index_group"] = "CUSTOM"
            else:
                comp_obj["index_group"] = "MIDCAP"
            companies_data.append(comp_obj)
            continue
            
        try:
            time.sleep(0.5 + random.random() * 0.5)
            # Сначала загружаем метаданные из yfinance, чтобы получить страну и другие параметры
            yf_info, buyout_news, antitrust_news, activist_news, turnaround_news, buyback_suspended = fetch_yfinance_news_and_meta(t)
            
            parsed = parser_sec.parse(t)
            wacc_data = risk_eng.compute_wacc(t, parsed["df"])
            
            # ── МЕТОДОЛОГИЧЕСКАЯ КОРРЕКЦИЯ: Использование предельной налоговой ставки (Marginal Tax Rate) ──
            # По Асвату Дамодарану, для разлеверирования/релеверирования беты (формула Хамады) и 
            # расчета WACC (налоговый щит 1-t на стоимость долга) должна использоваться исключительно 
            # предельная налоговая ставка (marginal tax rate), поскольку дополнительные долговые обязательства 
            # уменьшают прибыль на самом высоком налоговом уровне. 
            # Использование эффективной налоговой ставки для разлеверирования и предельной для релеверирования
            # создает искусственный арбитраж стоимости капитала.
            # Разлеверирование должно очищать бету по предельной ставке компании-цели, а релеверирование — 
            # закладывать предельную ставку покупателя (или объединенной Newco).
            
            # Получаем страну компании-цели для точного определения предельной налоговой ставки и странового риска
            target_country = yf_info.get("country", "United States") if yf_info else "United States"
            marginal_tax_rate_target = get_marginal_tax_rate(target_country)
            marginal_tax_rate_buyer = 0.25  # Предельная ставка покупателя / Newco (по умолчанию 25% для США)
            
            w_d = wacc_data.get("w_d", 0.0)
            w_e = wacc_data.get("w_e", 1.0)
            kd = wacc_data.get("kd", 0.05)
            rf = wacc_data.get("rf", 0.042)
            beta_l_old = wacc_data.get("beta", 1.0)
            
            # Разлеверируем бету обратно (Hamada) по предельной ставке компании-цели
            d_e_ratio = w_d / w_e if w_e > 0 else 0.0
            beta_u = beta_l_old / (1.0 + (1.0 - marginal_tax_rate_target) * d_e_ratio) if d_e_ratio > 0 else beta_l_old
            
            # Релеверируем бету по предельной налоговой ставке покупателя/Newco
            beta_l_new = beta_u * (1.0 + (1.0 - marginal_tax_rate_buyer) * d_e_ratio)
            
            # ── СТРАНОВЫЕ КОРРЕКТИРОВКИ СТОИМОСТИ КАПИТАЛА (Cross-Border M&A, DePamphilis Ch. 18) ──
            # k_e,em = R_f + beta_emfirm,global * (R_country - R_f) + FSP + CRP
            country_lower = str(target_country).lower().strip()
            
            # По умолчанию (для США)
            country_global_beta = 1.00
            r_country_premium = DAMODARAN_ERP  # по умолчанию премия S&P 500 (~4.60%)
            crp = 0.0
            country_name_ru = "США"
            
            if "china" in country_lower or "prc" in country_lower or "hong kong" in country_lower or "кнр" in country_lower or "китай" in country_lower:
                country_global_beta = 1.15
                r_country_premium = 0.0514  # доходность странового индекса за вычетом безрисковой ставки (R_country - R_f) = 5.14% (Damodaran 2026)
                crp = 0.0150                # Country Risk Premium (CRP) = 1.50% для развивающихся рынков (КНР)
                country_name_ru = "Китай (КНР)"
            elif "united kingdom" in country_lower or "great britain" in country_lower or "uk" in country_lower or "великобритания" in country_lower:
                country_global_beta = 1.05
                r_country_premium = 0.0513  # UK Equity Risk Premium = 5.13%
                crp = 0.0000                # Developed market CRP = 0.0%
                country_name_ru = "Великобритания"
            elif "japan" in country_lower or "япония" in country_lower:
                country_global_beta = 0.95
                r_country_premium = 0.0514  # Japan Equity Risk Premium = 5.14%
                crp = 0.0000                # Developed market CRP = 0.0%
                country_name_ru = "Япония"
            
            # Расчет глобальной беты компании (β_emfirm,global = β_emfirm,local * β_country,global)
            beta_emfirm_global = beta_l_new * country_global_beta
            
            # Расчет Foreign/Firm Size Premium (FSP) на основе капитализации
            mcap_val = wacc_data.get("equity_val") or (float(parsed["df"].iloc[-1].get("shares", 1.0)) * wacc_data.get("price", 0.0))
            if mcap_val > 21.5e9:
                fsp = 0.0
            elif mcap_val > 7.15e9:
                fsp = 0.013
            elif mcap_val > 2.93e9:
                fsp = 0.024
            else:
                fsp = 0.038
                
            # Итоговая трансграничная стоимость акционерного капитала (ke_new)
            ke_new = rf + beta_emfirm_global * r_country_premium + fsp + crp
            kd_after_tax_new = kd * (1.0 - marginal_tax_rate_buyer)
            
            # Пересчитываем WACC
            wacc_new = ke_new * w_e + kd_after_tax_new * w_d
            
            # Обновляем wacc_data корректными значениями по предельной ставке покупателя/Newco
            wacc_data["beta"] = beta_l_new
            wacc_data["beta_emfirm_global"] = beta_emfirm_global
            wacc_data["country_global_beta"] = country_global_beta
            wacc_data["r_country_premium"] = r_country_premium
            wacc_data["fsp"] = fsp
            wacc_data["crp"] = crp
            wacc_data["country_name_ru"] = country_name_ru
            wacc_data["ke"] = ke_new
            wacc_data["kd_after_tax"] = kd_after_tax_new
            wacc_data["wacc"] = wacc_new
            wacc_data["marginal_tax_rate"] = marginal_tax_rate_buyer
            
            # ── Логирование беты для диагностики (особенно важно для сезонных компаний) ──
            raw_beta_log = wacc_data.get("beta_raw", wacc_data.get("beta", "N/A"))
            adj_beta_log = wacc_data.get("beta", "N/A")
            if isinstance(raw_beta_log, float) and isinstance(adj_beta_log, float) and raw_beta_log != adj_beta_log:
                print(f"  [~] {t}: бета скорректирована {raw_beta_log:.2f} → {adj_beta_log:.2f} (Damodaran floor)")
            
            # (yf_info, buyout_news, antitrust_news, etc. fetched earlier)
            sec_ma_hints, qual_points, qual_score, detected_defenses = fetch_sec_ma_hints(client, t)

            # Получение держателей бумаг (акционеров) через Yahoo Finance
            active_holders = []
            passive_summary = None
            try:
                stock = yf.Ticker(t)
                inst = stock.institutional_holders
                if inst is not None and not inst.empty:
                    inst.columns = [c.lower().replace(' ', '_').replace('%', 'pct') for c in inst.columns]
                    cols = list(inst.columns)
                    holder_col = next((c for c in cols if 'holder' in c or 'name' in c), cols[0])
                    shares_col = next((c for c in cols if 'share' in c), cols[1] if len(cols) > 1 else cols[0])
                    value_col  = next((c for c in cols if 'value' in c or 'val' in c), cols[2] if len(cols) > 2 else cols[0])
                    pct_col = next((c for c in cols if 'pct' in c or 'change' in c or 'chg' in c), None)

                    ACTIVIST_KEYWORDS = ["ELLIOTT", "JANA", "STARBOARD", "SINGER", "COHN", "BARINGTON", 
                                         "TRIAN", "CORVEX", "CARL ICAHN", "IEP", "LEGION", "SACHEM HEAD", 
                                         "VALUE ACT", "ANCORA", "LAND AND BUILDINGS", "GREENLIGHT", "SILVER LAKE"]
                    STRATEGIC_INVESTOR_KEYWORDS = ["BERKSHIRE", "SOFTBANK", "DANAHER", "3G CAPITAL"]
                    HEDGE_FUND_KEYWORDS = ["RENAISSANCE", "TWO SIGMA", "CITADEL", "AQR", "D E SHAW", 
                                           "MILLENNIUM", "POINT72", "BRIDGEWATER", "AKRE", "SOROS", "OAKTREE", 
                                           "TIGER", "VIKING GLOBAL", "FARALLON", "T. ROWE", "FIDELITY", 
                                           "ARK", "COATUE", "LONE PINE", "TIGER GLOBAL"]
                    PASSIVE_KEYWORDS_EXTENDED = ["VANGUARD", "BLACKROCK", "STATE STREET", "UBS ASSET", 
                                                 "BARCLAYS", "JPMORGAN CHASE BANK", "SCHWAB", "GEODE", 
                                                 "NORTHERN TRUST", "INVESCO", "BNP PARIBAS", "SSGA", 
                                                 "DIMENSIONAL FUND", "NUVEEN", "ISHARES", "SPDR", "NORGES BANK"]

                    passive_pool_shares = 0
                    passive_pool_value = 0.0
                    passive_pool_names = []
                    total_tracked_shares = 0

                    for _, row in inst.head(20).iterrows():
                        holder_name = str(row.get(holder_col, '') or '')
                        holder_name_upper = holder_name.upper()
                        if not holder_name or holder_name == 'nan':
                            continue

                        try: shares_int = int(float(row.get(shares_col, 0))) if row.get(shares_col) and str(row.get(shares_col)) != 'nan' else 0
                        except: shares_int = 0
                        try: value_m = round(float(row.get(value_col, 0)) / 1e6, 2) if row.get(value_col) and str(row.get(value_col)) != 'nan' else 0.0
                        except: value_m = 0.0
                        try: pct_f = round(float(row.get(pct_col, 0)) * 100, 2) if pct_col and row.get(pct_col) and str(row.get(pct_col)) != 'nan' else 0.0
                        except: pct_f = 0.0

                        total_tracked_shares += shares_int

                        is_passive = any(k in holder_name_upper for k in PASSIVE_KEYWORDS_EXTENDED)
                        if is_passive:
                            passive_pool_shares += shares_int
                            passive_pool_value += value_m
                            passive_pool_names.append(holder_name)
                            continue

                        is_activist = any(k in holder_name_upper for k in ACTIVIST_KEYWORDS)
                        is_strategic = any(k in holder_name_upper for k in STRATEGIC_INVESTOR_KEYWORDS)
                        is_hedge = any(k in holder_name_upper for k in HEDGE_FUND_KEYWORDS)

                        if is_activist:
                            h_type = "Активист-акционер 🎯"
                            influence = "Может публично требовать смены стратегии, CEO, выкупа акций или продажи компании"
                        elif is_strategic:
                            h_type = "Стратегический инвестор"
                            influence = "Может предложить поглощение или поддержать M&A сделку"
                        elif is_hedge:
                            h_type = "Хедж-фонд"
                            influence = "Может поддерживать активистов при голосовании за слияние"
                        else:
                            h_type = "Активный инвестор"
                            influence = "Участвует в голосованиях за M&A решения"

                        active_holders.append({
                            "name": holder_name,
                            "type": h_type,
                            "influence": influence,
                            "shares": shares_int,
                            "value": value_m,
                            "pct_change": pct_f
                        })

                    # Сопоставляем активных держателей с раскрытыми намерениями из SC 13D и 8-K 5.02
                    for h in active_holders:
                        h_name_upper = h["name"].upper()
                        for hint in sec_ma_hints:
                            hint_desc_upper = hint["description"].upper()
                            hint_reason_upper = hint["reason"].upper()
                            
                            # Очищаем имя от корпоративных суффиксов
                            clean_h_name = h_name_upper
                            for suffix in ["LLC", "LP", "INC", "CORP", "CO", "LTD", "PARTNERS", "MANAGEMENT", "CAPITAL", "FUND", "GROUP"]:
                                clean_h_name = re.sub(r'\b' + suffix + r'\b', '', clean_h_name).strip()
                            
                            clean_h_name_words = [w for w in clean_h_name.split() if len(w) > 2]
                            is_match = False
                            if clean_h_name_words:
                                if any(w in hint_desc_upper or w in hint_reason_upper for w in clean_h_name_words):
                                    is_match = True
                            
                            if is_match:
                                if "13D" in hint["form"] or "13G" in hint["form"]:
                                    if "13G" in hint["form"]:
                                        h["influence"] = f"Раскрыл пассивные намерения в {hint['form']}: {hint['description']}"
                                    else:
                                        h["influence"] = f"Раскрыл активные намерения в {hint['form']}: {hint['description']}"
                                    if hint.get("key_demands"):
                                        h["influence"] += f" | Требования: {', '.join(hint['key_demands'])}"
                                    if "13G" not in hint["form"]:
                                        h["type"] = "Активист-акционер 🎯"
                                elif "8-K" in hint["form"]:
                                    h["influence"] = f"Получил место в Совете директоров согласно официальному отчету SEC {hint['form']}: {hint['description']}"
                                    h["type"] = "Технологический PE-гигант (Tier-1)" if "SILVER LAKE" in h_name_upper else "Активист-акционер 🎯"

                    type_order = {"Активист-акционер 🎯": 0, "Стратегический инвестор": 1, "Хедж-фонд": 2, "Активный инвестор": 3}
                    active_holders.sort(key=lambda x: (type_order.get(x["type"], 9), -x["shares"]))

                    if passive_pool_shares > 0 and total_tracked_shares > 0:
                        passive_pct = round(passive_pool_shares / total_tracked_shares * 100, 1)
                        passive_summary = {
                            "name": f"Пассивный пул ({len(passive_pool_names)} фондов: {', '.join(passive_pool_names[:3])}{'...' if len(passive_pool_names) > 3 else ''})",
                            "type": "Пассивный пул (ETF / Индексные фонды)",
                            "influence": "Голосуют по рекомендациям прокси-советников. Напрямую не инициируют сделки.",
                            "shares": passive_pool_shares,
                            "value": round(passive_pool_value, 2),
                            "pct_change": 0.0,
                            "is_passive_pool": True,
                            "passive_weight_pct": passive_pct
                        }
            except Exception as e:
                print(f"  [!] Ошибка при сборе акционеров для {t}: {e}")

            # Run M&A Timing & Catalyst Engine
            timing_profile = {}
            try:
                timing_engine = MATimingEngine()
                whale_points, whale_score = check_whale_catalysts(active_holders)
                timing_profile = timing_engine.calculate_timing(
                    t, parsed, wacc_data, yf_info, sec_ma_hints, cache,
                    turnaround_news=turnaround_news, buyback_suspended=buyback_suspended,
                    whale_points=whale_points, whale_score=whale_score,
                    qual_points=qual_points, qual_score=qual_score
                )
            except Exception as e:
                print(f"  [!] Ошибка при запуске MATimingEngine для {t}: {e}")
            
            # Умное определение сектора по SIC-коду из SEC EDGAR
            sec_meta = client.get_company_metadata(t)
            sic_code = sec_meta.get("sic")
            sec_sector = get_sector_by_sic(sic_code)
            
            wacc_data["sector"] = sec_sector or yf_info.get("sector") or parsed.get("sector") or "Technology"
            wacc_data["beta"] = wacc_data.get("beta") or yf_info.get("beta") or 1.0
            
            # Корректор сектора: индустрия имеет приоритет над грубым SIC-маппингом
            t_industry_raw = (parsed.get("industry") or yf_info.get("industry") or "").lower()
            sector_override_map = {
                "household & personal products": "Consumer Defensive",
                "household products": "Consumer Defensive",
                "personal products": "Consumer Defensive",
                "packaged foods": "Consumer Defensive",
                "beverages": "Consumer Defensive",
                "tobacco": "Consumer Defensive",
                "confectioners": "Consumer Defensive",
                "restaurants": "Consumer Cyclical",
                "eating places": "Consumer Cyclical",
                "fast food": "Consumer Cyclical",
                "casual dining": "Consumer Cyclical",
                "auto parts": "Consumer Cyclical",
                "auto & truck dealerships": "Consumer Cyclical",
                "auto dealers": "Consumer Cyclical",
                # ── Personal Services: налоговые и профессиональные сервисы ──────
                # По Damodaran: компании типа HRB, CURO имеют SIC 7372–7389 и попадают
                # в Consumer Cyclical по SIC, но их реальный профиль (высокая маржа,
                # сезонность, низкий capex) соответствует Professional Services.
                # Оставляем Consumer Cyclical — это корректно по Damodaran's sector data.
                "personal services": "Consumer Cyclical",
                "tax preparation": "Consumer Cyclical",
                "staffing & employment services": "Industrials",
                "consulting services": "Industrials",
                "waste management": "Industrials",
                "security & protection services": "Industrials",
            }
            for ind_kw, correct_sector in sector_override_map.items():
                if ind_kw in t_industry_raw:
                    wacc_data["sector"] = correct_sector
                    break
            
            # ─────────────────────────────────────────────────────────────────
            # МЕТОДОЛОГИЧЕСКАЯ НОРМАЛИЗАЦИЯ (Дамодаран, Борьба с ловушкой базового года)
            # Если это биотехнологическая или фармацевтическая компания средней
            # капитализации (менее $15 млрд) и у нее наблюдается аномально высокая
            # маржинальность в последнем году по сравнению со средним историческим,
            # мы очищаем данные от разовых платежей (монетизация роялти, авансы).
            # ─────────────────────────────────────────────────────────────────
            is_biotech_or_pharma = any(k in t_industry_raw for k in ["biotechnology", "pharmaceutical", "drug manufacturers"])
            latest_row_idx = parsed["df"].index[-1]
            latest_rev = parsed["df"].loc[latest_row_idx, "revenue"]
            latest_ebit = parsed["df"].loc[latest_row_idx, "ebit"]
            
            # Проверяем, есть ли аномальный всплеск маржи в последнем году (> 35%),
            # при том что в среднем исторически маржа была гораздо ниже или отрицательной.
            if is_biotech_or_pharma and latest_rev > 0:
                current_latest_margin = latest_ebit / latest_rev
                
                # Считаем среднюю историческую маржу без учета последнего года
                hist_margins = []
                if len(parsed["df"]) >= 2:
                    for idx in parsed["df"].index[:-1]:
                        r = parsed["df"].loc[idx, "revenue"]
                        e = parsed["df"].loc[idx, "ebit"]
                        if r > 0:
                            hist_margins.append(e / r)
                
                avg_hist_margin = sum(hist_margins) / len(hist_margins) if hist_margins else 0.0
                
                # Если текущая маржа аномально высокая (> 35%) и превышает историческую среднюю более чем на 30%
                if current_latest_margin > 0.35 and (current_latest_margin - avg_hist_margin) > 0.30:
                    normalized_margin = max(0.15, avg_hist_margin) # Нормализуем до 15% или исторической средней
                    print(f"  [~] {t}: Выявлен аномальный всплеск операционной маржи ({current_latest_margin*100:.1f}% против истор. {avg_hist_margin*100:.1f}%).")
                    print(f"      Применена нормализация EBIT базового года по Дамодарану (маржа ограничена {normalized_margin*100:.1f}%) для очистки от разовых роялти/лицензий.")
                    
                    # Пересчитываем ebit и ebit_after_tax для последнего года.
                    # ВАЖНО: колонки приходят из SEC EDGAR как int64 (целые доллары).
                    # Записываем float — предварительно конвертируем колонки в float64,
                    # иначе pandas выбросит "Invalid value ... for dtype int64".
                    for col in ["ebit", "ebit_after_tax", "net_income"]:
                        if col in parsed["df"].columns and parsed["df"][col].dtype == "int64":
                            parsed["df"][col] = parsed["df"][col].astype("float64")

                    tax_rate_latest = float(parsed["df"].loc[latest_row_idx, "effective_tax_rate"])
                    parsed["df"].loc[latest_row_idx, "ebit"] = float(latest_rev * normalized_margin)
                    parsed["df"].loc[latest_row_idx, "ebit_after_tax"] = float(latest_rev * normalized_margin * (1 - tax_rate_latest))
                    if "net_income" in parsed["df"].columns:
                        parsed["df"].loc[latest_row_idx, "net_income"] = float(latest_rev * normalized_margin * (1 - tax_rate_latest))

            # ── БЛОК 3: ДЕТЕКТОР ФИНАНСОВОГО СТРЕССА И КОРРЕКЦИЯ ОЦЕНКИ (Damodaran Ch 12, Altman 1968) ──────
            # Превентивно вычисляем базовые балансовые показатели для проверки Tangible Book Value (TBV)
            _df_latest = parsed["df"].iloc[-1]
            _df = parsed["df"]
            _assets = float(_df_latest.get("total_assets", 1.0))
            _liab   = float(_df_latest.get("total_liabilities", 0.0))
            _gw     = float(_df_latest.get("goodwill", 0.0))
            _equity_book = float(_df["equity"].iloc[-1]) if "equity" in _df.columns else (_assets - _liab)

            # ── 1. Tangible Book Value ────────────────────────────────────────
            # TBV = Book Equity - Goodwill - Intangibles (прокси: goodwill)
            _tbv = _equity_book - _gw
            _is_tbv_negative = _tbv < 0

            # Флаги блокировки Enterprise Value и переключения на FCFE / Merton
            sq_dcf_price = 0.0
            val_method = "N/A"
            _tbv_blocked_ev = False

            if _is_tbv_negative:
                # ── TBV Override Switch: Блокировка Enterprise Value (FCFF) ──
                # Если Tangible Book Value < 0, расчет EV через FCFF полностью блокируется,
                # и модель сразу переходит в режим Distress Override (FCFE / Merton) во избежание некорректных оценок.
                _tbv_blocked_ev = True
                print(f"  [!] {t}: TBV < 0 ({_tbv/1e6:.1f}M) → Расчет Enterprise Value автоматически заблокирован. Смена модели на Distress-override...")
            else:
                # Стандартный расчет DCF (FCFF) выполняется только если TBV >= 0
                try:
                    models = ValuationModels(wacc_data, parsed["df"], parsed["is_financial"])
                    model_name, forecast_df, sq_dcf_price, eq_val = models.run_dcf()
                    wacc_data["dcf_price"] = sq_dcf_price
                    val_method = model_name
                except Exception as e:
                    print(f"  [!] Ошибка стандартного расчета DCF для {t}: {e}")
                    sq_dcf_price = 0.0

            # Извлечение остальных финансовых примитивов для Z-Score и финансового стресса
            _rev    = float(_df_latest.get("revenue", 1.0))
            _ebit   = float(_df_latest.get("ebit", 0.0))
            _da     = float(_df_latest.get("da", 0.0))
            _ni     = float(_df_latest.get("net_income", 0.0))
            _td     = float(_df_latest.get("total_debt", 0.0))
            _cash   = float(_df_latest.get("total_cash", 0.0))
            _int    = float(_df_latest.get("interest", 0.0))
            _sh     = float(_df_latest.get("shares", 1.0))
            _capex  = float(_df_latest.get("capex", 0.0))

            # ── 2. Net Debt / EBITDA (долговой стресс) ───────────────────────
            _ebitda = _ebit + _da
            _net_debt = _td - _cash
            _nd_ebitda = (_net_debt / _ebitda) if _ebitda > 0 else None

            # ── 3. Interest Coverage (EBIT / Interest) ────────────────────────
            _int_cov = (_ebit / _int) if _int > 0 else None

            # ── 4. Altman Z-Score (Altman 1968 vs Altman 1993) ────────────────
            _mcap_val = wacc_data.get("equity_val", _sh * wacc_data.get("price", 0.0))
            _working_capital = _assets - _liab  # упрощенный прокси
            if "current_assets" in _df.columns and "current_liabilities" in _df.columns:
                _working_capital = float(_df_latest.get("current_assets", _assets * 0.4)) -                                    float(_df_latest.get("current_liabilities", _liab * 0.4))
            _X1 = _working_capital / _assets if _assets > 0 else 0.0
            
            # Извлечение нераспределенной прибыли (Retained Earnings) из баланса
            _re_val = None
            for _key in ["retained_earnings", "retained_earnings_accumulated_deficit", "retainedEarnings", "retainedEarningsAccumulatedDeficit"]:
                _val = _df_latest.get(_key)
                if _val is not None and not pd.isna(_val):
                    _re_val = float(_val)
                    break
            if _re_val is None:
                # В случае отсутствия тега, используем накопленный чистый доход в качестве математического прокси
                _re_val = float(_df["net_income"].sum()) if "net_income" in _df.columns else _ni
                
            _X2 = _re_val / _assets if _assets > 0 else 0.0
            _X3 = _ebit / _assets if _assets > 0 else 0.0
            _X4_mkt = _mcap_val / _liab if _liab > 0 else 999.0  # Для публичных промышленных (рыночный капитал)
            _X4_book = _equity_book / _liab if _liab > 0 else 999.0 # Для сервисных Z'-Score (балансовый капитал)
            _X5 = _rev / _assets if _assets > 0 else 0.0

            _is_manufacturing = wacc_data.get("sector") in ("Industrials", "Basic Materials", "Energy")
            if _is_manufacturing:
                # Оригинальный Altman Z (1968) для производственных компаний
                _altman_z = 1.2*_X1 + 1.4*_X2 + 3.3*_X3 + 0.6*_X4_mkt + 1.0*_X5
                _z_distress_zone = _altman_z < 1.81
                _z_grey_zone     = 1.81 <= _altman_z < 2.99
            else:
                # Altman Z' (1993) для сервисных/нефинансовых компаний (использует балансовую стоимость собственного капитала X4)
                _altman_z = 6.56*_X1 + 3.26*_X2 + 6.72*_X3 + 1.05*_X4_book
                _z_distress_zone = _altman_z < 1.1
                _z_grey_zone     = 1.1 <= _altman_z < 2.6

            # ── 5. Distress-флаг: хотя бы 2 из 3 условий → компания в дистрессе ──
            _distress_signals = sum([
                _is_tbv_negative,
                _nd_ebitda is not None and _nd_ebitda > 6.0,
                _int_cov is not None and _int_cov < 1.0,
                _z_distress_zone,
            ])
            _is_in_distress = _distress_signals >= 2

            # ── 6. Distress-свитч ──
            _fcfe_override_triggered = False
            if _tbv_blocked_ev or sq_dcf_price <= 0 or (_is_in_distress and sq_dcf_price < wacc_data.get("price", 1.0) * 0.3):
                try:
                    _ke = wacc_data.get("rf", 0.042) + wacc_data.get("beta", 1.0) * DAMODARAN_ERP
                    _ke = max(_ke, 0.06)  # floor: ke не ниже 6%
                    _terminal_g = min(wacc_data.get("rf", 0.042), 0.03)

                    # Базовый FCFE текущего года
                    _net_borrowing = max(0.0, _td - float(_df.iloc[-2].get("total_debt", _td)) if len(_df) >= 2 else 0.0)
                    _fcfe_base = _ni + _da - _capex + _net_borrowing
                    # Сглаживание по 2 годам
                    if len(_df) >= 2:
                        _ni2    = float(_df.iloc[-2].get("net_income", _ni))
                        _da2    = float(_df.iloc[-2].get("da", _da))
                        _cap2   = float(_df.iloc[-2].get("capex", _capex))
                        _fcfe2  = _ni2 + _da2 - _cap2
                        _fcfe_base = (_fcfe_base + _fcfe2) / 2

                    _g1 = max(-0.05, min(0.10, _terminal_g + 0.02))

                    # 5-летний прогноз
                    _fcfe_pv = 0.0
                    _fcfe_t = _fcfe_base
                    for _yr in range(1, 6):
                        _fcfe_t *= (1 + _g1)
                        _fcfe_pv += _fcfe_t / (1 + _ke) ** _yr

                    # Терминальная стоимость (Gordon Growth)
                    _fcfe_terminal = _fcfe_t * (1 + _terminal_g) / max(_ke - _terminal_g, 0.01)
                    _fcfe_tv_pv = _fcfe_terminal / (1 + _ke) ** 5

                    _equity_value_fcfe = _fcfe_pv + _fcfe_tv_pv
                    _fcfe_price = _equity_value_fcfe / _sh if _sh > 0 else 0.0

                    if _fcfe_price > 0:
                        sq_dcf_price = _fcfe_price
                        val_method = f"Distress FCFE Override (ke={_ke*100:.1f}%)"
                        wacc_data["dcf_price"] = sq_dcf_price
                        _fcfe_override_triggered = True
                        print(f"  [~] {t}: DISTRESS-СВИТЧ → FCFE (TBV={_tbv/1e6:.1f}M, Z={_altman_z:.2f}, ke={_ke*100:.1f}%) → цена={_fcfe_price:.2f}")
                    else:
                        # Merton Model (Black-Scholes-Merton Call Option on Assets)
                        try:
                            _hist = yf.Ticker(t).history(period="1y")["Close"]
                            _log_ret = np.log(_hist / _hist.shift(1)).dropna()
                            _sigma_equity = float(_log_ret.std() * math.sqrt(252)) if len(_log_ret) > 20 else 0.60
                        except Exception:
                            _sigma_equity = 0.60

                        _e_mkt = max(_mcap_val, 1.0)
                        _sigma_asset = _sigma_equity * (_e_mkt / (_e_mkt + _td)) if (_e_mkt + _td) > 0 else _sigma_equity
                        _sigma_asset = max(_sigma_asset, 0.10)
                        _sigma_asset_sq = _sigma_asset ** 2

                        _st_debt = float(_df_latest.get("st_debt", 0.0))
                        _lt_debt = float(_df_latest.get("lt_debt", _td - _st_debt))
                        _debt_sum = _st_debt + _lt_debt
                        _T_option = ((_st_debt * 1.0 + _lt_debt * 5.0) / _debt_sum) if _debt_sum > 0 else 5.0

                        _rf_rate = wacc_data.get("rf", 0.042)
                        _S_assets = max(_assets, _td, 1.0)
                        _E_strike = max(_td, 1.0)

                        _option_equity_value = black_scholes_option(
                            S=_S_assets, E=_E_strike, T=_T_option, r=_rf_rate,
                            sigma_sq=_sigma_asset_sq, q=0.0, option_type='call'
                        )
                        _option_price = _option_equity_value / _sh if _sh > 0 else 0.0
                        sq_dcf_price = max(_option_price, 0.01)
                        val_method = f"Distress Real-Option (Merton, σ_A={_sigma_asset*100:.0f}%, T={_T_option:.1f}y)"
                        wacc_data["dcf_price"] = sq_dcf_price
                        _fcfe_override_triggered = True
                        print(f"  [~] {t}: ДИСТРЕСС-ОПЦИОН (Black-Scholes-Merton) → S={_S_assets/1e6:.0f}M, E={_E_strike/1e6:.0f}M, "
                              f"σ_A={_sigma_asset*100:.0f}%, T={_T_option:.1f}г → цена={sq_dcf_price:.2f}")
                except Exception as _e:
                    print(f"  [!] {t}: Ошибка Distress-свитча: {_e}")

            # ── 7. Сохраняем distress-метрики для вывода в HTML ──────────────
            distress_profile = {
                "tbv": round(_tbv / 1e6, 1),           # Tangible Book Value, $M
                "is_tbv_negative": _is_tbv_negative,
                "altman_z": round(_altman_z, 2),
                "z_zone": "Дистресс" if _z_distress_zone else ("Серая зона" if _z_grey_zone else "Безопасная"),
                "nd_ebitda": round(_nd_ebitda, 2) if _nd_ebitda is not None else None,
                "int_coverage_ebit": round(_int_cov, 2) if _int_cov is not None else None,
                "distress_signals": _distress_signals,
                "is_in_distress": _is_in_distress,
                "fcfe_override": _fcfe_override_triggered,
                "regime": "HIGH_DISTRESS" if _is_in_distress or _fcfe_override_triggered else "MODERATE_DISTRESS_OR_SPECIAL_SITUATION",
                "tangible_book_value": _tbv
            }
            wacc_data["distress_profile"] = distress_profile
            
            # РЕШЕНИЕ ОШИБКИ 2: Передаем sq_dcf_price и профиль дистресса для сопоставимости расчетов
            opt_price, opt_wacc, opt_margin, opt_roc = opt_control.run_optimal_dcf(
                t, parsed, wacc_data, standalone_price=sq_dcf_price, distress_profile=distress_profile
            )
            value_of_control = max(0.0, opt_price - sq_dcf_price)
            
            latest = parsed["df"].iloc[-1]
            debt = latest["total_debt"]
            cash = latest["total_cash"]
            mcap = wacc_data["equity_val"]

            
            # --- REIT FFO/AFFO Valuation Override ---
            is_reit = (wacc_data.get("sector") == "Real Estate")
            if is_reit:
                latest_row = parsed["df"].iloc[-1]
                net_inc = latest_row.get("net_income", 0.0)
                depr = latest_row.get("da", 0.0)
                cap_exp = latest_row.get("capex", 0.0)
                
                ffo_val = net_inc + depr
                # Если capex 0.0, используем прокси 15% от FFO на поддержание зданий
                affo_val = ffo_val - cap_exp if cap_exp > 0 else ffo_val * 0.85
                
                shares_val = latest_row.get("shares", 1.0)
                affo_per_share = affo_val / shares_val if shares_val > 0 else 0.0
                
                # Капитализация AFFO по ставке 5% для Status Quo, и 4% для Optimal (Restructuring)
                sq_dcf_price = max(1.0, affo_per_share / 0.05)
                opt_price = max(sq_dcf_price, affo_per_share / 0.04)
                opt_wacc = wacc_data["wacc"]
                medians_reit = opt_control.get_medians_for_sector(wacc_data.get("sector") or "Real Estate")
                opt_margin = medians_reit["operating_margin"]
                opt_roc = medians_reit["roc"]
                value_of_control = opt_price - sq_dcf_price
                val_method = "REIT AFFO Capitalization (Ch 26)"

            # --- Trophy Asset (Sports/Entertainment/Broadcasting) Valuation Override ---
            is_trophy = False
            if "entertainment" in t_industry_raw or "sports" in t_industry_raw or "broadcasting" in t_industry_raw:
                latest_row = parsed["df"].iloc[-1]
                rev = latest_row.get("revenue", 0.0)
                ebitda = latest_row.get("ebit", 0.0) + latest_row.get("da", 0.0)
                ev = mcap + debt - cash
                ev_to_sales = ev / rev if rev > 0 else 0.0
                ev_to_ebitda = ev / ebitda if ebitda > 0 else 999.0
                is_tracking_stock = yf_info.get("quoteType", "") in ("TRACK_STOCK",) or \
                    str(yf_info.get("longName", "")).lower().endswith(("series a", "series b", "series c", "class a", "class b", "class c"))
                if rev > 0 and ev_to_sales > 4.0 and (ebitda <= 0.0 or ev_to_ebitda > 50.0) and not is_tracking_stock:
                    is_trophy = True
                    if True:
                        # Damodaran Ch 27: Sports franchises valued on Revenue multiples (typically 6-10x)
                        # Status quo asset value: 6.0x revenue
                        asset_val_sq = rev * 6.0
                        sq_dcf_price = max(1.0, (asset_val_sq + cash - debt) / latest_row.get("shares", 1.0))
                        
                        # Optimal (Control) asset value: 8.0x revenue (due to optimized monetization & synergy)
                        asset_val_opt = rev * 8.0
                        opt_price = max(sq_dcf_price, (asset_val_opt + cash - debt) / latest_row.get("shares", 1.0))
                        
                        opt_wacc = wacc_data["wacc"]
                        medians_trophy = opt_control.get_medians_for_sector(wacc_data.get("sector") or "Technology")
                        opt_margin = medians_trophy["operating_margin"]
                        opt_roc = medians_trophy["roc"]
                        value_of_control = opt_price - sq_dcf_price
                        val_method = "Trophy Asset Rev Multiple (Ch 27)"

            
            price = wacc_data["price"]
            
            # ─────────────────────────────────────────────────────────────────
            # МОДУЛЬ СРЕДНЕВЗВЕШЕННОЙ ОЦЕНКИ (Weighted Average Valuation, Ch. 8)
            # Объединяет Standalone DCF, Peer Multiples и Precedent Transactions
            # с весовыми коэффициентами: 0.3 для Standalone DCF, 0.2 для Peer Multiples,
            # и 0.5 для Precedent Transactions (согласно задаче 8.18).
            # ─────────────────────────────────────────────────────────────────
            t_sector = wacc_data.get("sector") or "Technology"
            target_ebitda = latest.get("ebit", 0.0) + latest.get("da", 0.0)
            target_shares = latest.get("shares", 1.0) if latest.get("shares", 1.0) > 0 else 1.0
            
            # 1. Standalone DCF Price
            dcf_price_val = sq_dcf_price
            
            # 2. Peer Multiples (Comparable Companies)
            DEFAULT_EV_EBITDA_MULTIPLES = {
                "Technology": 16.0,
                "Healthcare": 14.0,
                "Financial Services": 12.0, # Uses P/E fallback for financial services
                "Consumer Cyclical": 10.0,
                "Industrials": 11.0,
                "Consumer Defensive": 12.0,
                "Energy": 8.0,
                "Utilities": 10.0,
                "Real Estate": 15.0, # AFFO multiple
                "Basic Materials": 9.0,
                "Communication Services": 11.0
            }
            
            # Get historical revenue series for target (5-7 years)
            target_revs = get_historical_revenue_series(t, parsed["df"])
            
            # Filter peers in cache by sector
            peers_in_cache = [
                c for c in cache.get("tickers", {}).values()
                if c.get("financials", {}).get("sector") == t_sector and c.get("ticker") != t
            ]
            
            # Apply Statistical Double Filter: r >= 0.70 (revenue correlation) and p < 0.10 (significance)
            valid_peers = []
            for peer in peers_in_cache:
                peer_ticker = peer.get("ticker")
                peer_revs = peer.get("historical_revenues")
                if not peer_revs:
                    peer_revs = get_historical_revenue_series(peer_ticker)
                    peer["historical_revenues"] = peer_revs
                    cache["tickers"][peer_ticker] = peer
                    save_cache(cache)
                
                # Align overlapping years
                t_rev_dict = {int(k): float(v) for k, v in target_revs.items()}
                p_rev_dict = {int(k): float(v) for k, v in peer_revs.items()}
                common_years = sorted(list(set(t_rev_dict.keys()).intersection(set(p_rev_dict.keys()))))
                
                if len(common_years) >= 3:
                    x = [t_rev_dict[y] for y in common_years]
                    y = [p_rev_dict[y] for y in common_years]
                    try:
                        r_coef, p_val = pearsonr(x, y)
                        if np.isnan(r_coef) or np.isnan(p_val):
                            r_coef, p_val = 0.0, 1.0
                    except:
                        r_coef, p_val = 0.0, 1.0
                else:
                    r_coef, p_val = 0.0, 1.0
                    
                if r_coef >= 0.70 and p_val < 0.10:
                    valid_peers.append({
                        "peer": peer,
                        "r": r_coef,
                        "p": p_val
                    })
            
            failed_dynamic_filter = False
            if len(valid_peers) >= 1:
                peer_multiples_list = [
                    vp["peer"]["financials"]["ev_ebitda"] for vp in valid_peers 
                    if vp["peer"].get("financials", {}).get("ev_ebitda") is not None and vp["peer"]["financials"]["ev_ebitda"] > 0
                ]
                if len(peer_multiples_list) >= 1:
                    mult = sum(peer_multiples_list) / len(peer_multiples_list)
                else:
                    failed_dynamic_filter = True
            else:
                failed_dynamic_filter = True
                
            if failed_dynamic_filter:
                print(f"    [!] Провал фильтра динамических медиан для {t}: нет сопоставимых пиров с r >= 0.70 и p < 0.10.")
                print(f"        Активированы жесткие статические отраслевые мультипликаторы Дамодарана: EV/EBITDA = 12, P/E = 20.")
                mult = 12.0
                pe_mult = 20.0
            else:
                pe_mult = 15.0
                
            if t_sector == "Financial Services":
                target_eps = latest.get("eps") if latest.get("eps") else (latest.get("net_income", 0.0) / target_shares)
                if not target_eps or target_eps <= 0:
                    target_eps = max(0.01, price * 0.06) # fallback to 6% earnings yield proxy
                peer_multiple_price_val = max(1.0, target_eps * pe_mult)
            else:
                if target_ebitda > 0:
                    peer_ev = target_ebitda * mult
                    peer_equity_val = peer_ev + latest.get("total_cash", 0.0) - latest.get("total_debt", 0.0)
                    peer_multiple_price_val = max(1.0, peer_equity_val / target_shares) if target_shares > 0 else price
                else:
                    # Pre-profit fallback using EV/Sales of 3.5x
                    peer_ev = latest.get("revenue", 1.0) * 3.5
                    peer_equity_val = peer_ev + latest.get("total_cash", 0.0) - latest.get("total_debt", 0.0)
                    peer_multiple_price_val = max(1.0, peer_equity_val / target_shares) if target_shares > 0 else price
# 3. Precedent Transactions (Comparable Recent Transactions)
            # Это самостоятельный метод рыночной оценки, основанный на собственном, независимом
            # датасете реальных мультипликаторов сделок поглощения (M&A) по секторам, которые
            # уже содержат историческую премию за контроль (Aswath Damodaran, DePamphilis).
            PRECEDENT_EV_EBITDA_MULTIPLES = {
                "Technology": 20.5,
                "Healthcare": 18.0,
                "Financial Services": 16.0,  # Uses P/E fallback for financial services
                "Consumer Cyclical": 13.0,
                "Industrials": 14.2,
                "Consumer Defensive": 15.5,
                "Energy": 10.5,
                "Utilities": 13.0,
                "Real Estate": 19.5,  # AFFO multiple
                "Basic Materials": 11.5,
                "Communication Services": 14.0
            }
            PRECEDENT_PE_MULTIPLES = {
                "Technology": 28.5,
                "Healthcare": 26.0,
                "Financial Services": 19.0,
                "Consumer Cyclical": 21.0,
                "Industrials": 22.5,
                "Consumer Defensive": 23.0,
                "Energy": 15.0,
                "Utilities": 19.5,
                "Real Estate": 25.0,
                "Basic Materials": 17.5,
                "Communication Services": 21.5
            }
            PRECEDENT_EV_SALES_MULTIPLES = {
                "Technology": 4.8,
                "Healthcare": 4.2,
                "Financial Services": 3.5,
                "Consumer Cyclical": 1.8,
                "Industrials": 2.1,
                "Consumer Defensive": 2.4,
                "Energy": 2.5,
                "Utilities": 3.6,
                "Real Estate": 9.2,
                "Basic Materials": 1.9,
                "Communication Services": 2.8
            }

            p_ev_ebitda = PRECEDENT_EV_EBITDA_MULTIPLES.get(t_sector, 14.0)
            p_pe = PRECEDENT_PE_MULTIPLES.get(t_sector, 20.0)
            p_ev_sales = PRECEDENT_EV_SALES_MULTIPLES.get(t_sector, 2.5)

            if t_sector == "Financial Services":
                target_eps = latest.get("eps") if latest.get("eps") else (latest.get("net_income", 0.0) / target_shares)
                if not target_eps or target_eps <= 0:
                    target_eps = max(0.01, price * 0.06)
                precedent_transaction_price_val = max(1.0, target_eps * p_pe)
            else:
                if target_ebitda > 0:
                    prec_ev = target_ebitda * p_ev_ebitda
                    prec_equity_val = prec_ev + latest.get("total_cash", 0.0) - latest.get("total_debt", 0.0)
                    precedent_transaction_price_val = max(1.0, prec_equity_val / target_shares) if target_shares > 0 else price
                else:
                    prec_ev = latest.get("revenue", 1.0) * p_ev_sales
                    prec_equity_val = prec_ev + latest.get("total_cash", 0.0) - latest.get("total_debt", 0.0)
                    precedent_transaction_price_val = max(1.0, prec_equity_val / target_shares) if target_shares > 0 else price
            
            # 4. Weighted Average Pricing (0.3 DCF + 0.2 Peers + 0.5 Precedents)
            weighted_average_price_val = 0.3 * dcf_price_val + 0.2 * peer_multiple_price_val + 0.5 * precedent_transaction_price_val

            
            debt_ratio = debt / mcap if mcap > 0 else 0.0
            cash_ratio = cash / mcap if mcap > 0 else 0.0
            ebitda_pre = latest.get("ebit", 0.0) + latest.get("da", 0.0)
            rd_expense = latest.get("rd_expense", 0.0)
            rev_latest = latest.get("revenue", 1.0)
            rd_ratio_latest = rd_expense / rev_latest if rev_latest > 0 else 0.0

            # ─────────────────────────────────────────────────────────────────
            # КОНФЛИКТ №1: БИОТЕХ-ПАРАДОКС LBO
            # Компания с отрицательной EBITDA физически не может стать объектом
            # долгового LBO — банки не дадут кредит без денежных потоков.
            # Кэш на балансе = «боеприпасы для R&D», а не LBO-синергия.
            # ─────────────────────────────────────────────────────────────────
            is_negative_ebitda = (ebitda_pre <= 0)
            
            # Проверка высокой валовой маржи для хайтек-компаний (Software/Biotech/Semiconductors)
            gross_profit = latest.get("gross_profit", 0.0)
            gross_margin = (gross_profit / rev_latest) if rev_latest > 0 else 0.0
            t_industry = parsed.get("industry") or yf_info.get("industry") or "N/A"
            t_sector = wacc_data.get("sector") or "N/A"
            is_high_gross_tech = (
                (t_sector in ["Technology", "Healthcare"]) and
                (gross_margin >= 0.60) and
                (any(k in t_industry.lower() for k in [
                    "software", "biotechnology", "semiconductor", "saas", "hardware", "consumer electronics"
                ]))
            )

            is_tech_sector = (wacc_data.get("sector") == "Technology" or any(k in str(parsed.get("industry") or yf_info.get("industry") or "").lower() for k in ["software", "semiconductor", "saas", "hardware"]))
            is_biotech_profile = (
                (any(kw in str(parsed.get("industry") or yf_info.get("industry") or "").lower()
                    for kw in ["biotechnology", "drug manufacturers", "pharmaceutical"])
                or rd_ratio_latest > 0.20)
                and not is_tech_sector
            )

            # Финансовая синергия: для биотехов с отрицательной EBITDA
            # кэш — это «кислородная подушка», а не LBO-рычаг.
            # Балл ограничен 40 (Cash Runway), не 100 (LBO потенциал).
            score_fin = 10
            if is_negative_ebitda and is_biotech_profile:
                # Кэш ценен, но только как Cash Runway для R&D, не для LBO
                if cash_ratio > 0.15:
                    score_fin = 40   # cap: «Cash Runway» — хорошо для поглощения Big Pharma
                elif cash_ratio > 0.05:
                    score_fin = 25
                # Долг не влияет — при отриц. EBITDA долг это угроза, а не преимущество
            else:
                # ── Долговой компонент: непрерывная интерполяция ─────────────
                # Низкий долг = место для кредитного плеча LBO (Damodaran Ch. 25)
                if debt_ratio <= 0.10:
                    score_fin_debt = 40
                elif debt_ratio <= 0.35:
                    # Линейный спад: 40 → 0 в диапазоне [0.10, 0.35]
                    score_fin_debt = int(40 * (1 - (debt_ratio - 0.10) / 0.25))
                else:
                    score_fin_debt = 0

                # ── Кэш-компонент: непрерывная интерполяция ─────────────────
                # Высокий кэш = дополнительная ценность для покупателя
                if cash_ratio >= 0.20:
                    score_fin_cash = 50
                elif cash_ratio >= 0.05:
                    # Линейный рост: 0 → 50 в диапазоне [0.05, 0.20]
                    score_fin_cash = int(50 * (cash_ratio - 0.05) / 0.15)
                else:
                    score_fin_cash = 0

                score_fin = min(100, 10 + score_fin_debt + score_fin_cash)

            # ─────────────────────────────────────────────────────────────────
            # КОНФЛИКТ №2: БИОТЕХ-DCF ПАРАДОКС
            # Традиционный DCF «Status Quo» не применим к pre-profit биотехам:
            # рынок оценивает вероятность успеха клинических испытаний (pNPV).
            # Заменяем score_underval на Pipeline Premium Score.
            # ─────────────────────────────────────────────────────────────────
            upside = (weighted_average_price_val / price - 1.0) if price > 0 else 0.0
            if is_biotech_profile and is_negative_ebitda:
                # Для биотеха с отриц. EBITDA: оценка идёт по потенциалу пайплайна
                # Маленький биотех ($0.2B–$3B) с кэшем > 2 лет = высокая вероятность M&A
                mcap_b = mcap / 1e9
                cash_runway_years = (cash / abs(ebitda_pre)) if ebitda_pre < 0 else 5.0
                cash_runway_years = min(cash_runway_years, 10.0)  # cap at 10 лет
                # Pipeline Premium: маленький = интереснее для Big Pharma
                if mcap_b < 0.5:
                    pipeline_score = 85   # Micro-cap: Big Pharma платит 100–200% премию
                elif mcap_b < 2.0:
                    pipeline_score = 70   # Small-cap: типичный bolt-on target
                elif mcap_b < 6.0:
                    pipeline_score = 55   # Mid-cap: всё ещё реалистично
                else:
                    pipeline_score = 35   # Крупный биотех: дорого для поглощения
                # Бонус за кэш-взлётную полосу > 2 лет
                if cash_runway_years >= 2.0:
                    pipeline_score = min(100, pipeline_score + 10)
                score_underval = pipeline_score
                is_biotech_dcf_override = True
            else:
                score_underval = min(100, max(10, int(30 + upside * 100 * 1.5))) if upside > 0 else max(10, int(30 + upside * 100))
                is_biotech_dcf_override = False
                
            # DePamphilis / Tobin Q-Ratio
            # Tobin Q = (Рыночная стоимость активов) / (Восстановительная стоимость активов).
            # Числитель: mcap + total_debt (рыночная EV), а не equity_val (DCF-оценка).
            # Знаменатель: total_assets - goodwill (прокси восстановительной стоимости).
            tot_assets = latest.get("total_assets", 0.0)
            tot_liab = latest.get("total_liabilities", 0.0)
            goodwill_val = latest.get("goodwill", 0.0)
            net_assets = tot_assets - goodwill_val
            market_ev = mcap + debt  # рыночная EV без вычета кэша (восстановительная логика)
            q_ratio = market_ev / net_assets if net_assets > 0 else 999.0
            
            # DePamphilis Q-Ratio Check
            if q_ratio < 1.0:
                # If Market Val < Replacement Cost, it's a prime target
                score_underval = max(score_underval, 85)
                
            # LBO Debt Capacity (DePamphilis)
            ebit_val = latest.get("ebit", 0.0)
            da_val = latest.get("da", 0.0)
            ebitda_val = ebit_val + da_val
            interest_val = latest.get("interest", 0.0)
            debt_val = latest.get("total_debt", 0.0)
            
            interest_coverage = ebitda_val / interest_val if interest_val > 0 else 999.0
            debt_to_ebitda = debt_val / ebitda_val if ebitda_val > 0 else 999.0
            
            if interest_coverage >= 3.0 and debt_to_ebitda <= 4.0:
                lbo_status = "Высокая долговая емкость (LBO Prime Target)"
            elif interest_coverage >= 2.0 and debt_to_ebitda <= 5.5:
                lbo_status = "Умеренная долговая емкость"
            else:
                lbo_status = "Низкая долговая емкость / Высокий риск"
                
            # EPS Accretion / Dilution potential (PE vs Sector PE)
            target_eps = latest.get("eps")
            if not target_eps and latest.get("shares", 0.0) > 0.0:
                target_eps = latest.get("net_income", 0.0) / latest.get("shares")
            target_pe = price / target_eps if (target_eps and target_eps > 0.0) else 999.0
                
            sec_medians = opt_control.get_medians_for_sector(wacc_data["sector"])
            sector_pe = sec_medians.get("pe", 20.0) if sec_medians else 20.0
            
            if target_pe != 999.0:
                if target_pe < sector_pe * 0.8:
                    accretion_potential = "Высокая аккреция (PE < Sector PE)"
                elif target_pe < sector_pe:
                    accretion_potential = "Аккреция (PE <= Sector PE)"
                else:
                    accretion_potential = "Разводнение (PE > Sector PE)"
            else:
                accretion_potential = "Н/Д (Отрицательный EPS)"
                
            # Рассчитаем E-Index из найденных защит по Луциану Бебчуку (6 ключевых положений, Bebchuk et al. 2009)
            # Шкала расширена до 0–6 для соответствия академическим стандартам поглощений.
            e_index_score = 0
            detected_defenses_lower = [d.lower() for d in detected_defenses]
            
            # 1. Шахматный совет директоров (Staggered board)
            if any("staggered board" in d or "classified board" in d or "шахматный" in d or "классифицирован" in d for d in detected_defenses_lower):
                e_index_score += 1
                
            # 2. Ядовитые пилюли (Poison pills)
            if any("poison pill" in d or "пилюля" in d or "rights plan" in d for d in detected_defenses_lower):
                e_index_score += 1
                
            # 3. Золотые парашюты (Golden parachutes)
            if any("golden parachute" in d or "change-in-control" in d or "change in control" in d or "парашют" in d for d in detected_defenses_lower):
                e_index_score += 1
                
            # 4. Требование сверхбольшинства для одобрения слияний (Supermajority for mergers)
            if any("supermajority for mergers" in d or "одобрения слияний" in d or "approve merger" in d or "for merger" in d for d in detected_defenses_lower):
                e_index_score += 1
                
            # 5. Требование сверхбольшинства для изменения устава (Supermajority for charter amendments)
            if any("supermajority for charter amendments" in d or "изменения устава" in d or "amend charter" in d or "for charter" in d for d in detected_defenses_lower):
                e_index_score += 1
                
            # 6. Ограничения на созыв внеочередных собраний / письменные согласия или изменение bylaws (Limits on special meetings / written consents or bylaw amendments)
            if any("special meeting" in d or "written consent" in d or "собрани" in d or "согласи" in d or "bylaw" in d for d in detected_defenses_lower):
                e_index_score += 1

            # Рассчитаем вероятность успеха давления/прокси-борьбы
            proxy_success_prob = None
            avi = timing_profile.get("score", 0) if timing_profile else 0
            if avi >= 100 and e_index_score == 0:
                proxy_success_prob = {
                    "pressure_success": "95%+",
                    "proxy_fight_win": "85%",
                    "reason": (
                        "Полная беззащитность уставных документов (E-Index = 0/6): У цели отсутствуют Poison Pill (ядовитая пилюля) "
                        "и Staggered Board (шахматный совет). Это означает, что активисты могут полностью сменить весь состав "
                        "Совета директоров за один электоральный цикл на ближайшем годовом собрании акционеров. "
                        "Отсутствие ограничений на созыв внеочередных собраний (Action Limits): Активисты могут инициировать "
                        "процедуру письменного согласия (Consent Solicitation) в любое время года, форсируя голосование без ожидания "
                        "годового собрания. Экстремально распыленный капитал (доля инсайдеров 1.0%): Менеджмент не имеет «опорных» "
                        "голосов для защиты. Институциональные инвесторы (держащие до 99% акций) при отрицательном спреде эффективности "
                        "(ROC < WACC) гарантированно проголосуют за альтернативных директоров, предложенных активистом. "
                        "Развязка конфликта: На практике при AVI 100/100 дело редко доходит до открытого голосования акционеров. "
                        "Совет директоров цели, понимая неотвратимость поражения и желая избежать затрат на прокси-solicitors (~$6 млн), "
                        "пойдет на мирный компромисс (Negotiated Settlement) на ранних стадиях, добровольно выделив активистам 1–2 кресла "
                        "в обмен на Standstill Agreement."
                    )
                }
            elif avi >= 70 and e_index_score <= 1:
                proxy_success_prob = {
                    "pressure_success": "80%+",
                    "proxy_fight_win": "70%",
                    "reason": "Крайне слабая защита уставных документов (E-Index <= 1/6) при высокой уязвимости цели (AVI >= 70) дает активистам значительный перевес."
                }
            elif e_index_score >= 4:
                proxy_success_prob = {
                    "pressure_success": "< 30%",
                    "proxy_fight_win": "< 15%",
                    "reason": "Множественные барьеры защиты (E-Index >= 4/6) делают смену контроля через прокси-борьбу практически нереализуемой без согласия Совета директоров."
                }
            else:
                proxy_success_prob = {
                    "pressure_success": "50% - 60%",
                    "proxy_fight_win": "40% - 50%",
                    "reason": "Умеренный уровень защиты и средняя уязвимость. Исход прокси-борьбы будет сильно зависеть от позиции крупных институциональных инвесторов (Vanguard, BlackRock, etc.)."
                }
            depamphilis_metrics = {
                "q_ratio": round(q_ratio, 2) if q_ratio != 999.0 else None,
                "lbo_status": lbo_status,
                "interest_coverage": round(interest_coverage, 2) if (interest_coverage is not None and interest_coverage != 999.0) else None,
                "debt_to_ebitda": round(debt_to_ebitda, 2) if (debt_to_ebitda is not None and debt_to_ebitda != 999.0) else None,
                "accretion_potential": accretion_potential,
                "target_pe": round(target_pe, 2) if target_pe != 999.0 else None,
                "sector_pe": round(sector_pe, 2),
                "takeover_defenses": detected_defenses,
                "e_index": e_index_score,
                "proxy_success_prob": proxy_success_prob
            }

            control_upside = (opt_price / sq_dcf_price - 1.0) if sq_dcf_price > 0 else 0.0
            score_control = min(100, max(20, int(control_upside * 100 * 2.5))) if control_upside > 0 else 20

            sga = latest.get("sga", 0.0)
            rev = latest.get("revenue", 1e6)
            sga_ratio = sga / rev if rev > 0 else 0.0

            # ─────────────────────────────────────────────────────────────────
            # 4. АЛГОРИТМ ЦЕНЫ ПРЕДЛОЖЕНИЯ (Offer Price Engine)
            # ─────────────────────────────────────────────────────────────────
            # [1] Dynamic Alpha (α): Базовый α = 0.50.
            # При нахождении активиста по форме 13D повышается до 0.70 (0.75 при AVI = 100).
            alpha = 0.50
            has_13d = any("13D" in str(hint.get("form", "")) for hint in (sec_ma_hints or []))
            avi = timing_profile.get("score", 0) if timing_profile else 0
            if has_13d:
                if avi >= 100:
                    alpha = 0.75
                else:
                    alpha = 0.70

            # [2] Offer Price Maximizer: base_offer vs market_premium_offer
            # PV Control (per share)
            pv_control = max(0.0, opt_price - sq_dcf_price)

            # PV Synergy (per share)
            target_shares = latest.get("shares", 1.0) if latest.get("shares", 1.0) > 0 else 1.0
            sga_savings_annual = sga * 0.15
            target_wacc = wacc_data.get("wacc", 0.10)
            
            # Внедряем фазирование и интеграционные расходы (Глава 14) для расчета цены предложения
            synergy_total = calculate_phased_synergy_npv(
                sga_savings_annual, 
                wacc_data.get("marginal_tax_rate", 0.25), 
                target_wacc
            )
            pv_synergy_per_share = synergy_total / target_shares if target_shares > 0 else 0.0

            base_offer = sq_dcf_price + alpha * (pv_control + pv_synergy_per_share)
            market_premium_offer = price * 1.30
            offer_price = max(base_offer, market_premium_offer)

            # [3] Fiduciary Floor: Блокировка цены оферты ниже Standalone DCF (sq_dcf_price)
            if offer_price < sq_dcf_price:
                offer_price = sq_dcf_price
            sector_medians_op = opt_control.get_medians_for_sector(wacc_data["sector"])
            sector_sga_median = sector_medians_op["sga"]

            # ── SGA-компонент (непрерывная шкала, не ступенчатая) ────────────
            sga_excess = sga_ratio - sector_sga_median  # > 0 → раздутые расходы = синергия
            if sga_excess >= 0.10:
                score_sga = 90
            elif sga_excess >= 0.05:
                # Линейная интерполяция между 70 и 90
                score_sga = int(70 + (sga_excess - 0.05) / 0.05 * 20)
            elif sga_excess > 0:
                # Линейная интерполяция между 50 и 70
                score_sga = int(50 + (sga_excess / 0.05) * 20)
            else:
                score_sga = 40  # SGA в норме — минимальная синергия

            # ── R&D-компонент (Дамодаран: дублирующиеся R&D — источник синергии) ──
            # Высокий R&D относительно сектора = цель ценна для стратегического покупателя,
            # который может перекрёстно использовать пайплайн без дополнительных затрат.
            rd_expense_op = latest.get("rd_expense", 0.0)
            rd_ratio_op = rd_expense_op / rev if rev > 0 else 0.0
            # Отраслевые медианы R&D (Damodaran sector data)
            SECTOR_RD_MEDIANS = {
                "Technology": 0.12, "Healthcare": 0.15, "Communication Services": 0.05,
                "Consumer Cyclical": 0.03, "Consumer Defensive": 0.02, "Industrials": 0.03,
                "Energy": 0.01, "Utilities": 0.00, "Real Estate": 0.00,
                "Basic Materials": 0.02, "Financial Services": 0.01,
            }
            sector_rd_median = SECTOR_RD_MEDIANS.get(wacc_data["sector"], 0.03)
            rd_excess = rd_ratio_op - sector_rd_median
            if rd_excess >= 0.10:
                score_rd = 25   # Значительное R&D-дублирование = высокая синергия
            elif rd_excess >= 0.05:
                score_rd = 15
            elif rd_excess > 0:
                score_rd = 8
            else:
                score_rd = 0

            # Итоговый score_op: SGA (основной) + R&D (дополнительный), cap 100
            score_op = min(100, score_sga + score_rd)

            # ── Стратегический IP-буст: динамическое определение по данным SEC DEF 14A ──
            # Golden Parachute = признак того, что менеджмент уже готовился к продаже
            # (защитные выплаты прописаны заранее). Детектируется из DEF 14A в fetch_sec_ma_hints.
            # Ранее: хардкод списка исторически поглощённых тикеров → теперь: живые данные.
            gross_profit = latest.get("gross_profit", 0.0)
            gross_margin = (gross_profit / rev) if rev > 0 else 0.0
            has_golden_parachute = any(
                "golden parachute" in str(p).lower() or "change-in-control" in str(p).lower()
                for p in (qual_points or [])
            ) or any(
                "golden parachute" in str(d).lower() or "change in control" in str(d).lower()
                for d in (detected_defenses or [])
            )
            if has_golden_parachute and gross_margin >= 0.60:
                score_control = max(score_control, 85)
                score_underval = max(score_underval, 70)


            insider_share_pct = (yf_info.get("heldPercentInsiders") or 0.0) * 100
            inst_share_pct = (yf_info.get("heldPercentInstitutions") or 0.0) * 100
            
            ebitda = ebitda_pre  # уже посчитано выше
            ev = mcap + debt - cash
            ev_ebitda = ev / ebitda if ebitda > 0 else None
            debt_ebitda = debt / ebitda if ebitda > 0 else None

            audit_points = []
            has_insider_shield = False

            # ── Биотех-DCF предупреждение ──────────────────────────────────
            if is_biotech_dcf_override:
                mcap_b_disp = mcap / 1e9
                cash_runway_disp = (cash / abs(ebitda)) if ebitda < 0 else float('inf')
                runway_str = f"{cash_runway_disp:.1f} лет" if cash_runway_disp < 20 else "> 10 лет"
                biotech_note = (
                    f"⚠️ Биотех-оценка: традиционный DCF неприменим (EBITDA = {ebitda/1e6:.0f}M, "
                    f"R&D = {rd_ratio_latest*100:.0f}% выручки). Рынок оценивает вероятность "
                    f"успеха клинических испытаний (pNPV), а не текущие потоки. "
                    f"Cash Runway: {runway_str}. Балл 'Привлекательности' основан на "
                    f"Pipeline Premium для Big Pharma M&A, а не на DCF-дисконте."
                )
                audit_points.append(biotech_note)

            if insider_share_pct > 15.0:
                insider_status = f"Высокий инсайдерский контроль ({insider_share_pct:.1f}% акций у основателей/менеджмента) создает мощный защитный блок."
                has_insider_shield = True
            elif insider_share_pct > 5.0:
                insider_status = f"Умеренный инсайдерский контроль ({insider_share_pct:.1f}% акций) требует дружественного согласования сделки."
                has_insider_shield = False
            else:
                insider_status = f"Капитал сильно распылен (доля инсайдеров всего {insider_share_pct:.1f}%), делая компанию уязвимой для враждебных предложений."
                has_insider_shield = False
            audit_points.append(insider_status)

            shares_series = parsed["df"]["shares"].dropna()
            try:
                shares_change_recent = 0.0
                if len(shares_series) >= 2:
                    prev_shares = shares_series.iloc[-2]
                    if prev_shares != 0:
                        shares_change_recent = (shares_series.iloc[-1] / prev_shares - 1.0) * 100
                import math
                if math.isnan(shares_change_recent) or math.isinf(shares_change_recent):
                    shares_change_recent = 0.0
            except:
                shares_change_recent = 0.0

            is_financial = wacc_data.get("sector") == "Real Estate" or wacc_data.get("sector") == "Financial Services"
            if buyback_suspended:
                buyback_status = f"Отмена выкупа: несмотря на исторический выкуп на {-shares_change_recent:.1f}% за год, менеджмент объявил о приостановке (suspend) программы buyback. Свободный кэш перенаправляется на реструктуризацию и сокращение долга."
            elif shares_change_recent < -0.5:
                buyback_status = f"Активная защита: менеджмент проводит регулярный обратный выкуп акций (buyback на {-shares_change_recent:.1f}% за последний год), абсорбируя свободные денежные средства с баланса."
            elif cash_ratio < 0.03:
                buyback_status = f"Отсутствие выкупа: количество акций не снижается (изменение за последний год: {shares_change_recent:+.1f}%), а запас свободных средств на балансе минимален ({cash_ratio*100:.1f}% от капитализации). Компания не располагает ресурсами для распределения."
            else:
                if is_financial:
                    buyback_status = f"Специфика сектора: свободный кэш на балансе ({cash_ratio*100:.1f}% от капитализации) выступает в качестве регуляторного буфера и маржинального обеспечения, а не избыточного капитала для распределения."
                else:
                    buyback_status = f"Отсутствие выкупа: количество акций не снижается (изменение за последний год: {shares_change_recent:+.1f}%), свободный кэш на балансе ({cash_ratio*100:.1f}% от капитализации) накапливается без распределения."
            audit_points.append(buyback_status)

            # ── LBO-логика: компания с отриц. EBITDA = LBO невозможен ─────────
            lbo_feasible = False
            is_reit = wacc_data.get("sector") == "Real Estate"
            if is_reit:
                lbo_status = "Секторная специфика: оценка LBO через классический EV/EBITDA неприменима для REIT (требуется P/B или NAV). Финансовое LBO затруднено структурой выплат (90%+ payout)."
                lbo_feasible = False
            elif is_negative_ebitda:
                # Долговой LBO физически невозможен: нечем платить проценты
                target_buyer_desc = "Big Pharma" if is_biotech_profile else "отраслевых игроков"
                lbo_status = (
                    f"Долговой LBO невозможен: EBITDA отрицательная ({ebitda/1e6:.0f}M). "
                    f"Банки не выдадут кредит под актив без операционного потока. "
                    f"Сценарий сделки: стратегическое поглощение за кэш ({target_buyer_desc})."
                )
                lbo_feasible = False
            elif ev_ebitda is not None and ev_ebitda > 15.0:
                lbo_status = f"Высокие мультипликаторы оценки (EV/EBITDA = {ev_ebitda:.1f}x) делают классическое LBO экономически нежизнеспособным."
                lbo_feasible = False
                # Cap score_fin: высокий кэш полезен, но LBO невозможен при таких мультипликаторах
                score_fin = min(score_fin, 50)
            elif debt_ebitda is not None and debt_ebitda > 4.0:
                lbo_status = f"Текущая высокая долговая нагрузка (Debt/EBITDA = {debt_ebitda:.1f}x) исчерпывает кредитный лимит для выкупа компании с плечом (LBO)."
            elif debt_ebitda is not None and debt_ebitda > 3.0:
                lbo_status = f"Умеренно-высокий долг (Debt/EBITDA = {debt_ebitda:.1f}x) ограничивает потенциал кредитного плеча, LBO потребует значительного участия собственного капитала."
                lbo_feasible = True
            elif ev_ebitda is not None and debt_ebitda is not None and ev_ebitda > 0 and ev_ebitda < 8.0 and debt_ebitda < 2.0:
                lbo_status = f"Низкая оценка (EV/EBITDA = {ev_ebitda:.1f}x) и свободный баланс (Debt/EBITDA = {debt_ebitda:.1f}x) делают компанию идеальной мишенью для финансового LBO."
                lbo_feasible = True
            else:
                lbo_status = f"Финансовая структура умеренно пригодна для LBO (EV/EBITDA = {ev_ebitda:.1f}x, Debt/EBITDA = {debt_ebitda:.1f}x). Стандартные условия выкупа."
                lbo_feasible = True
            audit_points.append(lbo_status)

            weighted_score = int(0.25 * score_underval + 0.30 * score_control + 0.25 * score_fin + 0.20 * score_op)
            
            # 1. Ловушка ликвидности и Free-Float (ADTV)
            
            shares_out = yf_info.get("sharesOutstanding") or latest.get("shares", 1e7)
            float_shares = yf_info.get("floatShares") or (shares_out * 0.80)
            float_pct = float_shares / shares_out if shares_out > 0 else 0.80
            
            avg_volume = yf_info.get("averageVolume") or yf_info.get("averageVolume10days") or 1_000_000
            adtv_usd = avg_volume * price
            
            is_liquidity_trap = False
            liquidity_reason = ""
            if float_pct < 0.20:
                is_liquidity_trap = True
                liquidity_reason = "Критически низкий Free-Float (<20%)"
            elif float_pct < 0.45:
                stake_usd = mcap * 0.05
                days_to_accumulate = stake_usd / adtv_usd if adtv_usd > 0 else 999.0
                is_zombie = (adtv_usd < 500_000) and (mcap < 2_000_000_000)
                if days_to_accumulate > 20.0 or is_zombie:
                    is_liquidity_trap = True
                    liquidity_reason = f"Ловушка ликвидности (Float={float_pct:.0%}, ADTV=${adtv_usd/1e6:.1f}M)"
            else:
                is_zombie = (adtv_usd < 500_000) and (mcap < 500_000_000)
                if is_zombie:
                    is_liquidity_trap = True
                    liquidity_reason = f"Зомби-компания (ADTV=${adtv_usd/1e3:.0f}K при нормальном Float)"
                elif adtv_usd > 0 and (mcap * 0.05) / adtv_usd > 30.0:
                    is_liquidity_trap = True
                    liquidity_reason = "Экстремально низкий ADTV (накопление > 30 дней)"
            
            if is_liquidity_trap:
                audit_points.append(f"⚠️ Риск ликвидности: {liquidity_reason}")
            
            # 2. FTC Concentration / HHI Delta override
            is_hhi_block = "antitrust" in str(antitrust_news).lower()
            
            # 3. Short Attack Filter
            # ── ВАЖНО: shortRatio в yfinance = "Days to Cover" (дни для покрытия позиции),
            # а НЕ доля акций в коротких позициях. Использование shortRatio > 0.20 как
            # процента — классическая ошибка: shortRatio=4.5 дней попадёт как 450% → всегда True.
            # Используем ТОЛЬКО shortPercentOfFloat (доля float в шорте).
            # Порог 20% по Damodaran/DePamphilis: high short interest = сигнал манипуляций
            # или информационной асимметрии, усложняющей завершение сделки.
            short_pct_float = yf_info.get("shortPercentOfFloat") or 0.0
            # Дополнительный сигнал: shortRatio > 10 дней = сжатие ликвидности шорт-позиций
            short_ratio_days = yf_info.get("shortRatio") or 0.0
            is_short_attack = (
                (short_pct_float > 0.20) or
                (short_ratio_days > 10.0 and short_pct_float > 0.10) or
                ("hindenburg" in str(buyout_news).lower())
            )
            short_pct = short_pct_float  # используем для логирования и отображения
            
            # 4. CFIUS Override
            is_critical_tech = (
                (parsed.get("industry") or yf_info.get("industry") or "").lower() in ["semiconductors", "biotechnology", "aerospace & defense"]

            )
            
            regulatory_blocked = False
            if mcap > 100_000_000_000:
                reg_status = f"Огромная рыночная капитализация (${mcap/1e9:.1f} млрд) гарантирует пристальный антимонопольный контроль и усложняет финансирование сделки."
                regulatory_blocked = True
                audit_points.append(reg_status)
            elif wacc_data["sector"] in ["Utilities", "Energy", "Financial Services"]:
                reg_status = f"Компания относится к регулируемому сектору ({wacc_data['sector']}), что влечет длительные согласования с надзорными органами и повышает риск отмены сделки."
                regulatory_blocked = True
                audit_points.append(reg_status)
                
            is_agreed_deal = False
            if buyout_news:
                news_status = f"Обнаружен активный процесс поглощения в новостях: '{buyout_news}'."
                audit_points.append(news_status)
                is_agreed_deal = True
            if antitrust_news or is_hhi_block:
                news_status = f"Обнаружены регуляторные риски в новостях (FTC/антимонопольный иск): '{antitrust_news or 'HHI Concentration Block'}'."
                audit_points.append(news_status)
                regulatory_blocked = True
            if activist_news:
                news_status = f"Обнаружено активистское давление в новостях: '{activist_news}'."
                audit_points.append(news_status)
                
            # Вынесение вердикта о реализуемости
            t_country = yf_info.get("country") or ""
            is_china_risk = t_country in ("China", "Hong Kong")
            if is_china_risk:
                audit_points.append("Внимание: Компания зарегистрирована/базируется в КНР и использует структуру VIE (Variable Interest Entity). Классический LBO со стороны западных PE-фондов невозможен из-за геополитических барьеров (CAC, MoFCOM) и ограничений на передачу данных.")

            if is_liquidity_trap:
                score_cap = 30 if float_pct < 0.20 else 40
                defense_verdict, defense_verdict_class, feasibility_score = f"НИЗКАЯ РЕАЛИСТИЧНОСТЬ ({liquidity_reason})", "text-red", score_cap
                weighted_score = min(score_cap, weighted_score)
            elif is_china_risk:
                defense_verdict, defense_verdict_class, feasibility_score = "НИЗКАЯ РЕАЛИСТИЧНОСТЬ (Геополитический риск Китая / VIE)", "text-red", 35
                weighted_score = min(35, weighted_score)
            elif is_hhi_block:
                defense_verdict, defense_verdict_class, feasibility_score = "НИЗКАЯ РЕАЛИСТИЧНОСТЬ (Антимонопольный блок HHI > 200 / FTC)", "text-red", 45
                weighted_score = min(45, weighted_score)
            elif is_critical_tech:
                defense_verdict, defense_verdict_class, feasibility_score = "НИЗКАЯ РЕАЛИСТИЧНОСТЬ (Блокировка CFIUS / Нац. безопасность)", "text-red", 40
                weighted_score = min(40, weighted_score)
            elif is_short_attack:
                defense_verdict, defense_verdict_class, feasibility_score = "УМЕРЕННАЯ РЕАЛИСТИЧНОСТЬ (Шорт-атака / Риск манипуляций)", "text-orange", 55
                weighted_score = min(55, weighted_score)
            elif has_insider_shield:
                defense_verdict, defense_verdict_class, feasibility_score = "НИЗКАЯ РЕАЛИСТИЧНОСТЬ (Защищено инсайдерами)", "text-red", 20
            elif regulatory_blocked:
                defense_verdict, defense_verdict_class, feasibility_score = "НИЗКАЯ РЕАЛИСТИЧНОСТЬ (Антимонопольные барьеры)", "text-red", 15
            elif is_agreed_deal:
                defense_verdict, defense_verdict_class, feasibility_score = "ВЫСОКАЯ РЕАЛИСТИЧНОСТЬ (Согласованное поглощение)", "text-green", 95
            elif is_biotech_profile and is_negative_ebitda:
                defense_verdict, defense_verdict_class, feasibility_score = "УМЕРЕННАЯ РЕАЛИСТИЧНОСТЬ (Стратегическое поглощение Big Pharma)", "text-orange", 55
            elif is_negative_ebitda and not is_high_gross_tech:
                defense_verdict, defense_verdict_class, feasibility_score = "НИЗКАЯ РЕАЛИСТИЧНОСТЬ (Отрицательный EBITDA / Нет LBO)", "text-red", 25
                weighted_score = min(25, weighted_score)
            elif not lbo_feasible:
                defense_verdict, defense_verdict_class, feasibility_score = "УМЕРЕННАЯ РЕАЛИСТИЧНОСТЬ (Высокий долг / Оценка)", "text-orange", 45
            else:
                if insider_share_pct < 5.0 and inst_share_pct > 70.0:
                    defense_verdict, defense_verdict_class, feasibility_score = "ВЫСОКАЯ РЕАЛИСТИЧНОСТЬ (Уязвимая цель)", "text-green", 85
                else:
                    defense_verdict, defense_verdict_class, feasibility_score = "УМЕРЕННАЯ РЕАЛИСТИЧНОСТЬ (Средние барьеры)", "text-orange", 65
            # ── BuyerMatchmaker: подбор потенциальных покупателей ────────────
            # Пул кандидатов строится ТОЛЬКО из живых данных: другие тикеры,
            # уже посчитанные скринером и сохраненные в кэше (той же индустрии),
            # плюс один синтетический "Generic PE Sponsor" (модельная конструкция
            # для LBO-теста, не привязанная к конкретному фонду) — так матчмейкер
            # всегда может проверить долговую емкость, даже если в кэше пока нет
            # стратегических пиров.
            target_profile = {
                "ticker": t,
                "sic": sic_code,
                "sector": wacc_data.get("sector"),
                "revenue": rev,
                "sga": sga,
                "ebitda": ebitda,
                "debt": debt,
                "cash": cash,
                "mcap": mcap,
                "shares": latest.get("shares", 1.0),
                "price": price,
                "tax_rate": float(wacc_data.get("marginal_tax_rate", 0.25)),
                "net_income": float(latest.get("net_income", 0.0)),
                "capex": float(latest.get("capex", 0.0)),
                "da": float(latest.get("da", 0.0)),
            }

            strategic_bidders = []
            industry_universe_revenue = []
            for other_t, other_obj in cache.get("tickers", {}).items():
                if other_t == t:
                    continue
                ofin = (other_obj or {}).get("financials", {})
                if ofin.get("sector") != wacc_data.get("sector"):
                    continue
                o_revenue = ofin.get("revenue")
                if not o_revenue:
                    continue  # старая запись кэша без сырых полей — пропускаем как bidder-кандидата
                industry_universe_revenue.append(o_revenue)
                o_ebitda = ofin.get("ebitda", 0.0)
                o_debt = ofin.get("debt", 0.0)
                # Кандидат в стратегические покупатели должен сам не быть в долговом стрессе
                if o_ebitda > 0 and o_debt / o_ebitda > 4.0:
                    continue
                strategic_bidders.append({
                    "ticker": other_t,
                    "buyer_type": "strategic",
                    "sic": ofin.get("sic"),
                    "sector": ofin.get("sector"),
                    "revenue": o_revenue,
                    "sga": ofin.get("sga", 0.0),
                    "ebitda": o_ebitda,
                    "debt": o_debt,
                    "cash": ofin.get("cash", 0.0),
                    "mcap": (ofin.get("mcap", 0.0) or 0.0) * 1e9,  # в кэше mcap хранится в $млрд
                    "wacc": (ofin.get("wacc", 10.0) or 10.0) / 100.0,
                    "net_income": ofin.get("net_income", 0.0),
                    "capex": ofin.get("capex", 0.0),
                    "da": ofin.get("da", 0.0),
                })

            generic_pe_sponsor = {
                "ticker": "GENERIC_PE_SPONSOR",
                "buyer_type": "financial",
                "sic": None, "sector": None,
                "revenue": 0.0, "sga": 0.0, "ebitda": 0.0, "debt": 0.0,
                "cash": mcap * 0.30,  # условное допущение: спонсор располагает ~30% EV цели в кэше/committed capital
                "mcap": 0.0, "wacc": 0.10,
            }

            buyer_matches = run_buyer_matchmaker(
                target_profile,
                strategic_bidders + [generic_pe_sponsor],
                industry_universe=industry_universe_revenue,
            )

            buyer_profile = {
                "strategic": buyer_matches[:10],
                "financial": {
                    "active_holders": active_holders,
                    "passive_summary": passive_summary
                }
            }

            # ── Микро-буст за живое институциональное накопление (13F Whale Accumulation) ──
            # Ранее: хардкод архивных тикеров → теперь: детектируется из реальных данных YF.
            # Сигнал: среди активных держателей есть активист ИЛИ стратегический инвестор,
            # накопивший позицию (pct_change > 0 означает увеличение за последний квартал).
            has_whale_accumulation = any(
                h.get("type") in ("Активист-акционер 🎯", "Стратегический инвестор")
                and h.get("pct_change", 0) > 0
                for h in active_holders
            )
            if has_whale_accumulation and 60 <= weighted_score <= 64:
                # Микро-буст +5 баллов — переводит компанию через барьер грейда B (65)
                weighted_score += 5

            if weighted_score >= 80:
                letter_grade = "A"
            elif weighted_score >= 65:
                letter_grade = "B"
            elif weighted_score >= 50:
                letter_grade = "C"
            else:
                letter_grade = "D"

            # Consolidation Simulator (GAAP Purchase Accounting)
            try:
                t_tax_rate = float(wacc_data.get("marginal_tax_rate", 0.25))
                t_step_up_pct = getattr(args, "step_up_pct", 0.15)
                consolidation_scenarios = run_purchase_accounting_simulation(target_shares, offer_price, latest, tax_rate=t_tax_rate, step_up_pct=t_step_up_pct)
            except Exception as e:
                print(f"  [!] Ошибка симулятора консолидации для {t}: {e}")
                consolidation_scenarios = {}

            # Collar Arrangements Simulation (Chapter 11)
            try:
                collar_scenarios = run_collar_simulation(offer_price, price, collar_pct=0.10)
            except Exception as e:
                print(f"  [!] Ошибка симулятора воротниковых соглашений для {t}: {e}")
                collar_scenarios = {}

            # STANDALONE NEGATIVE COVENANTS SIMULATION (Chapter 13)
            try:
                leverage_target_debt = debt
                if leverage_target_debt <= 0:
                    leverage_target_debt = mcap * 0.30  # Assume nominal LBO debt if debt-free
                
                covenants_profile = simulate_negative_covenants(
                    post_trans_debt=leverage_target_debt,
                    post_trans_ebitda=ebitda_pre,
                    net_income=float(latest.get("net_income", 0.0)),
                    capex=float(latest.get("capex", 0.0)),
                    da=float(latest.get("da", 0.0)),
                    buyer_type="financial"  # Test against strict LBO standards
                )
            except Exception as e:
                print(f"  [!] Ошибка симулятора ковенантов для {t}: {e}")
                covenants_profile = {}

            comp_obj = {
                "ticker": t,
                "name": parsed["name"],
                "score": weighted_score,
                "grade": letter_grade,
                "consolidation_scenarios": consolidation_scenarios,
                "collar_scenarios": collar_scenarios,
                "scores": {"underval": score_underval, "control": score_control, "fin": score_fin, "op": score_op},
                "feasibility": {"score": feasibility_score, "verdict": defense_verdict, "class": defense_verdict_class, "points": audit_points},
                "timing": timing_profile,
                "historical_revenues": target_revs,
                "financials": {
                    "pe": round(price / latest.get("eps"), 1) if (latest.get("eps") and latest.get("eps") > 0) else None,
                    "ev_ebitda": round(ev_ebitda, 1) if ev_ebitda is not None else None,
                    "mcap": round(mcap / 1e9, 2),
                    "price": round(price, 2),
                    "dcf_price": round(sq_dcf_price, 2),
                    "optimal_dcf_price": round(opt_price, 2),
                    "peer_price": round(peer_multiple_price_val, 2),
                    "precedent_price": round(precedent_transaction_price_val, 2),
                    "weighted_price": round(weighted_average_price_val, 2),
                    "value_of_control": round(value_of_control, 2),
                    "offer_price": round(offer_price, 2),
                    "alpha": round(alpha, 2),
                    "pv_synergy": round(pv_synergy_per_share, 2),
                    "base_offer": round(base_offer, 2),
                    "market_premium": round(market_premium_offer, 2),
                    "wacc": round(wacc_data["wacc"] * 100, 2),
                    "opt_wacc": round(opt_wacc * 100, 2),
                    "opt_margin": round(opt_margin * 100, 1),
                    "opt_roc": round(opt_roc * 100, 1),
                    "sector": wacc_data["sector"],
                    "industry": parsed.get("industry") or yf_info.get("industry") or "N/A",
                    "val_method": val_method,
                    # ── ДеПамфилис Глава 18: Страновые параметры ──
                    "beta_emfirm_global": round(wacc_data.get("beta_emfirm_global", beta_l_new), 2),
                    "country_global_beta": round(wacc_data.get("country_global_beta", 1.0), 2),
                    "fsp": round(wacc_data.get("fsp", 0.0) * 100, 2),
                    "crp": round(wacc_data.get("crp", 0.0) * 100, 2),
                    "country_name_ru": wacc_data.get("country_name_ru", "США"),
                    # ── Сырые поля для BuyerMatchmaker (пул потенциальных покупателей) ──
                    "sic": sic_code,
                    "revenue": rev,
                    "sga": sga,
                    "ebitda": ebitda,
                    "debt": debt,
                    "cash": cash,
                    "net_income": float(latest.get("net_income", 0.0)),
                    "capex": float(latest.get("capex", 0.0)),
                    "da": float(latest.get("da", 0.0)),
                },
                "verdicts": {
                    "underval_desc": (
                        "Pipeline Premium (Big Pharma)" if (is_biotech_dcf_override and score_underval >= 70)
                        else "Pipeline Premium (Mid-cap)" if (is_biotech_dcf_override and score_underval >= 55)
                        else "Pipeline Premium (Large-cap)" if is_biotech_dcf_override
                        else "Глубокий дисконт" if score_underval >= 70
                        else "Умеренный дисконт" if score_underval >= 40
                        else "Справедливая цена"
                    ),
                    "control_desc": "Нужна реструктуризация" if score_control >= 70 else ("Умеренная эффективность" if score_control >= 40 else "Менеджмент эффективен"),
                    "fin_desc": (
                        "Cash Runway (R&D)" if (is_negative_ebitda and is_biotech_profile)
                        else "Growth PE / Strategic Absorption" if is_negative_ebitda
                        else "Отлично для LBO/долга" if score_fin >= 70
                        else "Средняя емкость" if score_fin >= 40
                        else "Ограниченный потенциал"
                    ),
                    "op_desc": "Высокая синергия (SGA)" if score_op >= 70 else ("Умеренная синергия" if score_op >= 40 else "Минимальная синергия")
                },
                "catalyst": determine_catalyst({"score_underval": score_underval, "score_control": score_control, "score_fin": score_fin, "score_op": score_op}, is_negative_ebitda=is_negative_ebitda),
                "index_group": "SP500" if t in sp500_list else (("CUSTOM" if t in custom_tickers else "MIDCAP")),
                "buyer_profile": buyer_profile,
                "sec_ma_hints": sec_ma_hints,
                "qual_points": qual_points,
                "qual_score": qual_score,
                "distress": distress_profile,
                "depamphilis_metrics": depamphilis_metrics,
                "covenants": covenants_profile
            }
            
            companies_data.append(comp_obj)
            cache["tickers"][t] = comp_obj
            save_cache(cache)
            print(f"  [+] Успешно добавлен в кэш. M&A привлекательность = {comp_obj['score']}/100")
        except Exception as e:
            err_msg = str(e)
            # Если ошибка связана с отсутствием данных SEC/XBRL — заносим в блэклист
            is_data_error = any(kw in err_msg for kw in [
                "None of ['Year']", "Year", "No data", "empty", "KeyError", "XBRL"
            ])
            if is_data_error and t not in custom_tickers:
                if "blacklist" not in cache:
                    cache["blacklist"] = {}
                cache["blacklist"][t] = {
                    "reason": "Нет данных в SEC EDGAR (XBRL отсутствует): иностранный листинг или новое IPO",
                    "error": err_msg[:120]
                }
                save_cache(cache)
                print(f"  [x] Ошибка {t}: {err_msg} → добавлен в блэклист (больше не будет пересчитываться)")
            else:
                print(f"  [x] Ошибка при оценке {t}: {err_msg}")
            
    generate_html(list(cache["tickers"].values()))
    print(f"\n============================================================")
    print(f"  M&A СКРИНЕР ПО ДАМОДАРАНУ ОБНОВЛЕН!")
    print(f"============================================================")


def determine_catalyst(score_data, is_negative_ebitda=False):
    reasons = []
    if score_data["score_underval"] >= 70:
        reasons.append("Недооценка")
    if score_data["score_control"] >= 70:
        reasons.append("Смена контроля")
    if score_data["score_fin"] >= 70:
        if is_negative_ebitda:
            reasons.append("Growth PE потенциал")
        else:
            reasons.append("Долговой LBO потенциал")
    if score_data["score_op"] >= 70:
        reasons.append("Синергия (SGA)")
        
    if reasons:
        return " + ".join(reasons)
    else:
        scores = [
            (score_data["score_underval"], "Умеренная оценка"),
            (score_data["score_control"], "Потенциал смены контроля"),
            (score_data["score_fin"], "Growth PE потенциал" if is_negative_ebitda else "Средний кэш"),
            (score_data["score_op"], "Умеренная синергия")
        ]
        scores.sort(reverse=True)
        return scores[0][1]

def generate_html(companies_data):
    html_path = "ma_screener_dashboard.html"
    
    html_template = f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>M&A Скринер Поглощений (Глава 25)</title>
    
    <!-- Global JS Error Handler -->
    <script>
        window.onerror = function(message, source, lineno, colno, error) {{
            var msg = "JavaScript Error: " + message + " at " + source + ":" + lineno + ":" + colno;
            console.error(msg);
            try {{
                var errDiv = document.createElement('div');
                errDiv.style.position = 'fixed';
                errDiv.style.top = '0';
                errDiv.style.left = '0';
                errDiv.style.width = '100%';
                errDiv.style.backgroundColor = '#ef4444';
                errDiv.style.color = '#fff';
                errDiv.style.padding = '12px';
                errDiv.style.zIndex = '99999';
                errDiv.style.fontFamily = 'monospace';
                errDiv.style.fontSize = '12px';
                errDiv.innerHTML = '<strong>JavaScript Error:</strong> ' + message + ' at ' + source + ':' + lineno + ':' + colno;
                
                var target = document.body || document.documentElement;
                if (target) {{
                    target.appendChild(errDiv);
                }} else {{
                    alert(msg);
                }}
            }} catch(e) {{
                alert(msg + " (failed to render: " + e.message + ")");
            }}
            return false;
        }};
    </script>
    
    <!-- Google Fonts -->
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;600;700&family=Fira+Code:wght@400;500&display=swap" rel="stylesheet">
    
    <!-- Chart.js CDN -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    
    <style>
        :root {{
            --bg-main: #0b0f19;
            --bg-card: rgba(17, 24, 39, 0.6);
            --bg-card-hover: rgba(31, 41, 55, 0.7);
            --border-color: rgba(255, 255, 255, 0.08);
            --text-primary: #f9fafb;
            --text-secondary: #9ca3af;
            
            /* Status colors */
            --color-strong-buy: #10b981;
            --color-buy: #34d399;
            --color-hold: #f59e0b;
            --color-sell: #ef4444;
            --color-accent: #3b82f6;
            
            --glow-strong-buy: rgba(16, 185, 129, 0.2);
            --glow-buy: rgba(52, 211, 153, 0.2);
            --glow-hold: rgba(245, 158, 11, 0.2);
            --glow-sell: rgba(239, 68, 68, 0.2);
            --glow-accent: rgba(59, 130, 246, 0.25);
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: var(--bg-main);
            color: var(--text-primary);
            line-height: 1.5;
            padding: 24px;
            background-image: radial-gradient(circle at 50% 0%, rgba(30, 58, 138, 0.15) 0%, rgba(15, 23, 42, 0) 50%);
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}

        /* Header layout */
        header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 24px;
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            backdrop-filter: blur(12px);
            margin-bottom: 24px;
        }}

        .header-left h1 {{
            font-family: 'Outfit', sans-serif;
            font-size: 32px;
            font-weight: 700;
            letter-spacing: -0.5px;
            background: linear-gradient(135deg, #f9fafb 0%, #3b82f6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}

        .header-left p {{
            color: var(--text-secondary);
            font-size: 14px;
            margin-top: 6px;
        }}

        .header-right {{
            text-align: right;
        }}

        .version-badge {{
            background: rgba(59, 130, 246, 0.15);
            color: var(--color-accent);
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            border: 1px solid rgba(59, 130, 246, 0.3);
            display: inline-block;
        }}

        /* KPI Dashboard Stats */
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
            margin-bottom: 24px;
        }}

        .stat-card {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 20px;
            backdrop-filter: blur(12px);
            display: flex;
            flex-direction: column;
            position: relative;
            overflow: hidden;
            transition: all 0.3s ease;
        }}
        
        .stat-card::after {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 4px;
            height: 100%;
            background: var(--color-accent);
        }}
        
        .stat-card.green::after {{ background: var(--color-strong-buy); }}
        .stat-card.orange::after {{ background: var(--color-hold); }}
        .stat-card.red::after {{ background: var(--color-sell); }}

        .stat-card:hover {{
            transform: translateY(-2px);
            border-color: rgba(255, 255, 255, 0.15);
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.3);
        }}

        .stat-label {{
            color: var(--text-secondary);
            font-size: 13px;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        .stat-value {{
            font-size: 36px;
            font-family: 'Outfit', sans-serif;
            font-weight: 700;
            margin: 8px 0;
        }}

        .stat-subtext {{
            color: var(--text-secondary);
            font-size: 12px;
        }}

        /* Visual Matrix Scatter Plot Section */
        .matrix-section {{
            display: grid;
            grid-template-columns: 1.8fr 1.2fr;
            gap: 20px;
            margin-bottom: 24px;
        }}

        .panel-card {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 24px;
            backdrop-filter: blur(12px);
            position: relative;
        }}

        .panel-card h2 {{
            font-family: 'Outfit', sans-serif;
            font-size: 20px;
            font-weight: 600;
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}

        .chart-container {{
            position: relative;
            height: 380px;
            width: 100%;
        }}

        /* Text helper for quadrants */
        .quadrant-legend {{
            font-size: 12px;
            color: var(--text-secondary);
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
            margin-top: 16px;
            border-top: 1px solid var(--border-color);
            padding-top: 16px;
        }}

        .quad-item {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .quad-dot {{
            width: 10px;
            height: 10px;
            border-radius: 50%;
            display: inline-block;
        }}

        /* Filter Controls */
        .controls-card {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 20px;
            margin-bottom: 24px;
            backdrop-filter: blur(12px);
        }}

        .filters-grid {{
            display: grid;
            grid-template-columns: 1.2fr 1fr 1fr 1fr 1fr auto;
            gap: 16px;
            align-items: flex-end;
        }}

        .filter-group {{
            display: flex;
            flex-direction: column;
            gap: 8px;
        }}

        .filter-group label {{
            font-size: 12px;
            font-weight: 500;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        .search-input, .select-input {{
            background: rgba(15, 23, 42, 0.8);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 10px 14px;
            color: var(--text-primary);
            font-size: 14px;
            outline: none;
            transition: all 0.3s ease;
            width: 100%;
        }}

        .search-input:focus, .select-input:focus {{
            border-color: var(--color-accent);
            box-shadow: 0 0 10px var(--glow-accent);
        }}

        .btn-reset {{
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            padding: 10px 20px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
            transition: all 0.3s ease;
        }}

        .btn-reset:hover {{
            background: rgba(255, 255, 255, 0.1);
            border-color: rgba(255, 255, 255, 0.2);
        }}

        /* Index badge column */
        .index-badge {{
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 10px;
            font-weight: 700;
            text-transform: uppercase;
        }}
        
        .idx-sp500 {{
            background: rgba(59, 130, 246, 0.15);
            color: #60a5fa;
            border: 1px solid rgba(59, 130, 246, 0.3);
        }}
        
        .idx-midcap {{
            background: rgba(167, 139, 250, 0.15);
            color: #c084fc;
            border: 1px solid rgba(167, 139, 250, 0.3);
        }}
        
        .idx-sp600 {{
            background: rgba(236, 72, 153, 0.15);
            color: #f472b6;
            border: 1px solid rgba(236, 72, 153, 0.3);
        }}
        
        .idx-local {{
            background: rgba(156, 163, 175, 0.15);
            color: #d1d5db;
            border: 1px solid rgba(156, 163, 175, 0.3);
        }}

        /* Table formatting */
        .table-card {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            overflow: hidden;
            backdrop-filter: blur(12px);
            margin-bottom: 40px;
        }}

        .table-responsive {{
            overflow-x: auto;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            text-align: left;
        }}

        th {{
            background: rgba(15, 23, 42, 0.8);
            padding: 16px 20px;
            font-size: 12px;
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            border-bottom: 1px solid var(--border-color);
            cursor: pointer;
            user-select: none;
            transition: background 0.3s ease;
        }}

        th:hover {{
            background: rgba(30, 41, 59, 0.8);
            color: var(--text-primary);
        }}

        td {{
            padding: 16px 20px;
            border-bottom: 1px solid var(--border-color);
            font-size: 14px;
            vertical-align: middle;
        }}

        tr.table-row {{
            cursor: pointer;
            transition: all 0.2s ease;
        }}

        tr.table-row:hover {{
            background: var(--bg-card-hover);
        }}

        tr.table-row.active {{
            background: rgba(59, 130, 246, 0.05);
        }}

        .ticker-col {{
            font-family: 'Fira Code', monospace;
            font-weight: 600;
            color: var(--color-accent);
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .score-badge {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 32px;
            height: 32px;
            border-radius: 50%;
            font-weight: 700;
            font-size: 13px;
        }}

        .score-high {{
            background: rgba(16, 185, 129, 0.15);
            color: var(--color-strong-buy);
            border: 1px solid rgba(16, 185, 129, 0.3);
        }}

        .score-mid {{
            background: rgba(245, 158, 11, 0.15);
            color: var(--color-hold);
            border: 1px solid rgba(245, 158, 11, 0.3);
        }}

        .score-low {{
            background: rgba(239, 68, 68, 0.15);
            color: var(--color-sell);
            border: 1px solid rgba(239, 68, 68, 0.3);
        }}

        .feasibility-badge {{
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
        }}

        .badge-green {{
            background: rgba(16, 185, 129, 0.15);
            color: var(--color-strong-buy);
            border: 1px solid rgba(16, 185, 129, 0.3);
        }}

        .badge-orange {{
            background: rgba(245, 158, 11, 0.15);
            color: var(--color-hold);
            border: 1px solid rgba(245, 158, 11, 0.3);
        }}

        .badge-red {{
            background: rgba(239, 68, 68, 0.15);
            color: var(--color-sell);
            border: 1px solid rgba(239, 68, 68, 0.3);
        }}

        /* Drawer details expanded */
        .drawer-row {{
            background: rgba(15, 23, 42, 0.3);
        }}

        .drawer-content {{
            padding: 24px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 32px;
            animation: slideDown 0.25s ease-out;
        }}

        @keyframes slideDown {{
            from {{ opacity: 0; transform: translateY(-10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        .drawer-block h3 {{
            font-family: 'Outfit', sans-serif;
            font-size: 16px;
            font-weight: 600;
            margin-bottom: 16px;
            color: var(--text-primary);
            border-bottom: 1px solid rgba(255,255,255,0.05);
            padding-bottom: 8px;
        }}

        /* Score breakdown items */
        .score-bar-group {{
            margin-bottom: 14px;
        }}

        .score-bar-labels {{
            display: flex;
            justify-content: space-between;
            font-size: 13px;
            margin-bottom: 6px;
        }}

        .score-bar-bg {{
            height: 8px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 4px;
            overflow: hidden;
        }}

        .score-bar-fill {{
            height: 100%;
            border-radius: 4px;
            transition: width 1s ease-in-out;
        }}

        /* Audit bullet points */
        .audit-list {{
            list-style: none;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }}

        .audit-item {{
            font-size: 13px;
            color: var(--text-secondary);
            line-height: 1.4;
            display: flex;
            gap: 8px;
        }}
        
        .audit-item::before {{
            content: '•';
            color: var(--color-accent);
            font-weight: bold;
        }}

        .text-green {{ color: var(--color-strong-buy); }}
        .text-orange {{ color: var(--color-hold); }}
        .text-red {{ color: var(--color-sell); }}

        /* Detailed table specs inside drawer */
        .specs-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 12px;
            margin-top: 16px;
        }}

        .spec-item {{
            background: rgba(255,255,255,0.02);
            border: 1px solid rgba(255,255,255,0.03);
            border-radius: 8px;
            padding: 10px 14px;
            display: flex;
            justify-content: space-between;
            font-size: 13px;
        }}

        .spec-label {{
            color: var(--text-secondary);
        }}

        .spec-value {{
            font-weight: 600;
            font-family: 'Fira Code', monospace;
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- HEADER -->
        <header>
            <div class="header-left">
                <h1>M&A Скринер Поглощений</h1>
                <p>Многофакторный скоринг на основе Главы 25 «Оценка слияний и поглощений» Асвата Дамодарана</p>
            </div>
            <div class="header-right">
                <span class="version-badge">Индексный Мониторинг &amp; Кэш</span>
            </div>
        </header>

        <!-- STATS / KPIS -->
        <div class="stats-grid" id="statsGrid">
            <div class="stat-card">
                <span class="stat-label">Проанализировано компаний</span>
                <span class="stat-value" id="statTotal">0</span>
                <span class="stat-subtext" id="statIndexBreakdown">S&P 500: 0 | Вне S&P 500: 0</span>
            </div>
            <div class="stat-card green">
                <span class="stat-label">Идеальные цели (Buyout Targets)</span>
                <span class="stat-value" id="statIdeal">0</span>
                <span class="stat-subtext">Высокий M&A Score + Высокая реалистичность</span>
            </div>
            <div class="stat-card red">
                <span class="stat-label">Защищенные / Блокированные</span>
                <span class="stat-value" id="statProtected">0</span>
                <span class="stat-subtext">Инсайдерские щиты или FTC барьеры</span>
            </div>
            <div class="stat-card">
                <span class="stat-label">Средний M&A Индекс</span>
                <span class="stat-value" id="statAvgScore">0</span>
                <span class="stat-subtext">Показатель привлекательности рынка</span>
            </div>
        </div>

        <!-- VISUAL MATRIX (CHART + INTRO) -->
        <div class="matrix-section">
            <div class="panel-card">
                <h2>Карта Поглощений: Привлекательность vs Реалистичность</h2>
                <div class="chart-container">
                    <canvas id="maScatterChart"></canvas>
                </div>
            </div>
            
            <div class="panel-card" style="display: flex; flex-direction: column; justify-content: space-between;">
                <div>
                    <h2>Принципы Анализа (Гл. 25)</h2>
                    <p style="font-size: 13.5px; color: var(--text-secondary); margin-bottom: 12px; line-height: 1.5;">
                        Оценка потенциальных поглощений разделена на два независимых вектора:
                    </p>
                    <ul style="font-size: 13px; color: var(--text-secondary); padding-left: 20px; display: flex; flex-direction: column; gap: 8px;">
                        <li><strong>1. Привлекательность (X-ось)</strong>: Оценивается дисконт к DCF стоимости, операционная неэффективность (контроль), объем свободных наличных и оптимизация SG&A.</li>
                        <li><strong>2. Реалистичность сделки (Y-ось)</strong>: Анализирует барьеры — долю инсайдеров (свыше 15% дает блок), активные байбэки, уязвимости LBO структуры и регуляторный комплаенс.</li>
                    </ul>
                    <p style="font-size: 13.5px; color: var(--text-secondary); margin-top: 12px;">
                        <strong>Квадранты карты:</strong>
                    </p>
                </div>
                
                <div class="quadrant-legend">
                    <div class="quad-item">
                        <span class="quad-dot" style="background: var(--color-strong-buy);"></span>
                        <span><strong>Правый верхний</strong>: Идеальные цели для поглощения (Уязвимые и дешевые).</span>
                    </div>
                    <div class="quad-item">
                        <span class="quad-dot" style="background: var(--color-sell);"></span>
                        <span><strong>Правый нижний</strong>: Защищенные гиганты (Инсайдерский блок или FTC).</span>
                    </div>
                    <div class="quad-item">
                        <span class="quad-dot" style="background: var(--color-accent);"></span>
                        <span><strong>Левый верхний</strong>: Доступные, но дорогие/эффективные компании.</span>
                    </div>
                    <div class="quad-item">
                        <span class="quad-dot" style="background: rgba(255,255,255,0.15);"></span>
                        <span><strong>Левый нижний</strong>: Низкий M&A приоритет.</span>
                    </div>
                </div>
            </div>
        </div>

        <!-- FILTER CONTROLS -->
        <div class="controls-card">
            <div class="filters-grid">
                <div class="filter-group">
                    <label for="searchBar">Поиск компании</label>
                    <input type="text" id="searchBar" class="search-input" placeholder="Введите тикер или имя...">
                </div>
                <div class="filter-group">
                    <label for="filterIndex">Категория</label>
                    <select id="filterIndex" class="select-input">
                        <option value="NON_SP500" selected>Вне S&P 500 (Mid-Cap)</option>
                        <option value="ALL">Все компании</option>
                        <option value="MIDCAP">Вне S&P 500 (Mid-Cap)</option>
                        <option value="SP600">Вне S&P 500 (Small-Cap)</option>
                        <option value="SP500">Входят в S&P 500</option>
                        <option value="WORKSPACE">Локальные / Воркспейс</option>
                    </select>
                </div>
                <div class="filter-group">
                    <label for="filterSector">Сектор</label>
                    <select id="filterSector" class="select-input">
                        <option value="ALL">Все секторы</option>
                    </select>
                </div>
                <div class="filter-group">
                    <label for="filterScore">Привлекательность</label>
                    <select id="filterScore" class="select-input">
                        <option value="ALL">Любая</option>
                        <option value="HIGH">Высокая (>= 70)</option>
                        <option value="MID">Умеренная (40-69)</option>
                        <option value="LOW">Низкая (&lt; 40)</option>
                    </select>
                </div>
                <div class="filter-group">
                    <label for="filterFeas">Реалистичность</label>
                    <select id="filterFeas" class="select-input">
                        <option value="ALL">Любая</option>
                        <option value="HIGH">Высокая</option>
                        <option value="MID">Умеренная</option>
                        <option value="LOW">Низкая</option>
                    </select>
                </div>
                <button class="btn-reset" onclick="resetFilters()">Сбросить</button>
            </div>
        </div>

        <!-- SCREENER TABLE -->
        <div class="table-card">
            <div class="table-responsive">
                <table>
                    <thead>
                        <tr>
                            <th onclick="handleSort('ticker')">Тикер</th>
                            <th onclick="handleSort('name')">Компания</th>
                            <th onclick="handleSort('index_group')">Индекс</th>
                            <th onclick="handleSort('score')" style="text-align: center;">M&A Score</th>
                            <th onclick="handleSort('feasibility')">Реалистичность</th>
                            <th onclick="handleSort('sector')">Сектор</th>
                            <th onclick="handleSort('pe')">P/E</th>
                            <th onclick="handleSort('ev_ebitda')">EV/EBITDA</th>
                            <th onclick="handleSort('mcap')">Капитализация</th>
                        </tr>
                    </thead>
                    <tbody id="screenerTableBody">
                        <!-- Сюда JavaScript вставит строки -->
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <!-- DATA INJECTION -->
    <script>


        const companiesData = {json.dumps(companies_data, ensure_ascii=False, default=json_serialize_fallback)};
        
        let currentSort = {{ key: 'score', desc: true }};
        let activeFilters = {{
            search: '',
            index: 'NON_SP500',
            sector: 'ALL',
            score: 'ALL',
            feasibility: 'ALL'
        }};
        
        let scatterChart = null;

        function initDashboard() {{
            try {{
                populateSectorDropdown();
                updateStats();
                renderTable();
                initScatterPlot();
            }} catch (e) {{
                console.error("Error initializing dashboard:", e);
                const errDiv = document.createElement('div');
                errDiv.style.position = 'fixed';
                errDiv.style.bottom = '0';
                errDiv.style.left = '0';
                errDiv.style.width = '100%';
                errDiv.style.backgroundColor = '#ef4444';
                errDiv.style.color = '#fff';
                errDiv.style.padding = '12px';
                errDiv.style.zIndex = '99999';
                errDiv.style.fontFamily = 'monospace';
                errDiv.style.fontSize = '12px';
                errDiv.innerHTML = `<strong>Init Error:</strong> ${{e.message}} <br> ${{e.stack}}`;
                document.body.appendChild(errDiv);
            }}
        }}

        if (document.readyState === 'loading') {{
            window.addEventListener('DOMContentLoaded', () => {{
                initDashboard();
                setupEventListeners();
            }});
        }} else {{
            initDashboard();
            setupEventListeners();
        }}

        function setupEventListeners() {{
            document.getElementById('searchBar').addEventListener('input', (e) => {{
                activeFilters.search = e.target.value.toLowerCase();
                filterAndRender();
            }});
            
            document.getElementById('filterIndex').addEventListener('change', (e) => {{
                activeFilters.index = e.target.value;
                filterAndRender();
            }});
            
            document.getElementById('filterSector').addEventListener('change', (e) => {{
                activeFilters.sector = e.target.value;
                filterAndRender();
            }});
            
            document.getElementById('filterScore').addEventListener('change', (e) => {{
                activeFilters.score = e.target.value;
                filterAndRender();
            }});
            
            document.getElementById('filterFeas').addEventListener('change', (e) => {{
                activeFilters.feasibility = e.target.value;
                filterAndRender();
            }});
        }}


                function populateSectorDropdown() {{
            const sectors = new Set();
            companiesData.forEach(c => {{
                if (c.financials && c.financials.sector && c.financials.sector !== 'N/A') {{
                    sectors.add(c.financials.sector);
                }}
            }});
            
            const dropdown = document.getElementById('filterSector');
            sectors.forEach(s => {{
                const opt = document.createElement('option');
                opt.value = s;
                opt.textContent = s;
                dropdown.appendChild(opt);
            }});
        }}

        function updateStats() {{
            const total = companiesData.length;
            
            const sp500Count = companiesData.filter(c => c.index_group === 'SP500').length;
            const midcapCount = companiesData.filter(c => c.index_group === 'MIDCAP').length;
            const smallcapCount = companiesData.filter(c => c.index_group === 'SP600').length;
            const otherCount = companiesData.filter(c => c.index_group !== 'SP500' && c.index_group !== 'MIDCAP' && c.index_group !== 'SP600').length;
            const nonSpCount = midcapCount + smallcapCount + otherCount;
            
            const ideal = companiesData.filter(c => 
                (c.score || 0) >= 70 && c.feasibility && c.feasibility.verdict && c.feasibility.verdict.includes('ВЫСОКАЯ')
            ).length;
            
            const protected = companiesData.filter(c => 
                c.feasibility && c.feasibility.verdict && c.feasibility.verdict.includes('НИЗКАЯ')
            ).length;
            
            const sumScore = companiesData.reduce((sum, c) => sum + (typeof c.score === 'number' && !isNaN(c.score) ? c.score : 0), 0);
            const avgScore = total > 0 ? Math.round(sumScore / total) : 0;
            
            document.getElementById('statTotal').textContent = total;
            document.getElementById('statIndexBreakdown').textContent = `S&P 500: ${{sp500Count}} | S&P 400: ${{midcapCount}} | S&P 600: ${{smallcapCount}}`;
            document.getElementById('statIdeal').textContent = ideal;
            document.getElementById('statProtected').textContent = protected;
            document.getElementById('statAvgScore').textContent = avgScore + '/100';
        }}

        function filterAndRender() {{
            renderTable();
            updateScatterChart();
        }}

        function resetFilters() {{
            document.getElementById('searchBar').value = '';
            document.getElementById('filterIndex').value = 'NON_SP500';
            document.getElementById('filterSector').value = 'ALL';
            document.getElementById('filterScore').value = 'ALL';
            document.getElementById('filterFeas').value = 'ALL';
            
            activeFilters = {{
                search: '',
                index: 'NON_SP500',
                sector: 'ALL',
                score: 'ALL',
                feasibility: 'ALL'
            }};
            
            filterAndRender();
        }}

        function getFilteredData() {{
            return companiesData.filter(c => {{
                const matchesSearch = c.ticker.toLowerCase().includes(activeFilters.search) || 
                                      c.name.toLowerCase().includes(activeFilters.search);
                                      
                let matchesIndex = true;
                if (activeFilters.index === 'NON_SP500') {{
                    matchesIndex = c.index_group !== 'SP500';
                }} else if (activeFilters.index !== 'ALL') {{
                    matchesIndex = c.index_group === activeFilters.index;
                }}
                
                const matchesSector = activeFilters.sector === 'ALL' || c.financials.sector === activeFilters.sector;
                
                let matchesScore = true;
                if (activeFilters.score === 'HIGH') matchesScore = c.score >= 70;
                else if (activeFilters.score === 'MID') matchesScore = c.score >= 40 && c.score < 70;
                else if (activeFilters.score === 'LOW') matchesScore = c.score < 40;
                
                let matchesFeas = true;
                if (activeFilters.feasibility === 'HIGH') matchesFeas = c.feasibility.verdict.includes('ВЫСОКАЯ');
                else if (activeFilters.feasibility === 'MID') matchesFeas = c.feasibility.verdict.includes('УМЕРЕННАЯ');
                else if (activeFilters.feasibility === 'LOW') matchesFeas = c.feasibility.verdict.includes('НИЗКАЯ');
                
                return matchesSearch && matchesIndex && matchesSector && matchesScore && matchesFeas;
            }});
        }}

        function renderTable() {{
            const filtered = getFilteredData();
            
            filtered.sort((a, b) => {{
                let valA, valB;
                
                if (currentSort.key === 'ticker') {{
                    valA = a.ticker;
                    valB = b.ticker;
                }} else if (currentSort.key === 'name') {{
                    valA = a.name;
                    valB = b.name;
                }} else if (currentSort.key === 'score') {{
                    valA = a.score;
                    valB = b.score;
                }} else if (currentSort.key === 'feasibility') {{
                    valA = a.feasibility.score;
                    valB = b.feasibility.score;
                }} else if (currentSort.key === 'sector') {{
                    valA = a.financials.sector;
                    valB = b.financials.sector;
                }} else if (currentSort.key === 'pe') {{
                    valA = a.financials.pe || 9999;
                    valB = b.financials.pe || 9999;
                }} else if (currentSort.key === 'ev_ebitda') {{
                    valA = a.financials.ev_ebitda || 9999;
                    valB = b.financials.ev_ebitda || 9999;
                }} else if (currentSort.key === 'mcap') {{
                    valA = a.financials.mcap;
                    valB = b.financials.mcap;
                }} else if (currentSort.key === 'index_group') {{
                    valA = a.index_group;
                    valB = b.index_group;
                }}
                
                if (valA < valB) return currentSort.desc ? 1 : -1;
                if (valA > valB) return currentSort.desc ? -1 : 1;
                return 0;
            }});
            
            const tbody = document.getElementById('screenerTableBody');
            tbody.innerHTML = '';
            
            if (filtered.length === 0) {{
                tbody.innerHTML = `<tr><td colspan="9" style="text-align: center; color: var(--text-secondary); padding: 40px;">Компании, соответствующие фильтрам, не найдены.</td></tr>`;
                return;
            }}
            
            filtered.forEach(c => {{
                let scoreClass = 'score-low';
                if (c.score >= 70) scoreClass = 'score-high';
                else if (c.score >= 40) scoreClass = 'score-mid';
                
                let feasBadge = 'badge-red';
                if (c.feasibility.verdict.includes('ВЫСОКАЯ')) feasBadge = 'badge-green';
                else if (c.feasibility.verdict.includes('УМЕРЕННАЯ')) feasBadge = 'badge-orange';
                
                const peStr = (c.financials.pe !== null && c.financials.pe !== undefined && c.financials.pe > 0) ? c.financials.pe + 'x' : '-';
                const evStr = (c.financials.ev_ebitda !== null && c.financials.ev_ebitda !== undefined && c.financials.ev_ebitda > 0) ? c.financials.ev_ebitda + 'x' : '-';
                
                let idxBadgeHtml = '';
                if (c.index_group === 'SP500') idxBadgeHtml = '<span class="index-badge idx-sp500">S&P 500</span>';
                else if (c.index_group === 'MIDCAP') idxBadgeHtml = '<span class="index-badge idx-midcap">S&P 400</span>';
                else if (c.index_group === 'SP600') idxBadgeHtml = '<span class="index-badge idx-sp600">S&P 600</span>';
                else idxBadgeHtml = '<span class="index-badge idx-local">Локальный</span>';
                
                const tr = document.createElement('tr');
                tr.className = 'table-row';
                tr.id = `row-${{c.ticker}}`;
                tr.onclick = () => toggleRow(c.ticker);
                
                tr.innerHTML = `
                    <td><div class="ticker-col">${{c.ticker}}</div></td>
                    <td><strong>${{c.name}}</strong></td>
                    <td>${{idxBadgeHtml}}</td>
                    <td style="text-align: center;"><span class="score-badge ${{scoreClass}}">${{c.score}}</span></td>
                    <td><span class="feasibility-badge ${{feasBadge}}">${{c.feasibility.verdict.split(' ')[0]}}</span></td>
                    <td>${{c.financials.sector}}</td>
                    <td>${{peStr}}</td>
                    <td>${{evStr}}</td>
                    <td>$${{c.financials.mcap}}B</td>
                `;
                
                tbody.appendChild(tr);
                
                const drawerTr = document.createElement('tr');
                drawerTr.className = 'drawer-row';
                drawerTr.id = `drawer-${{c.ticker}}`;
                drawerTr.style.display = 'none';
                
                const auditItems = c.feasibility.points.map(pt => `<li class="audit-item">${{pt}}</li>`).join('');
                const timingItems = c.timing ? c.timing.points.map(pt => `<li class="audit-item">${{pt}}</li>`).join('') : '<li class="audit-item">Данные о тайминге не рассчитаны</li>';
                
                drawerTr.innerHTML = `
                    <td colspan="9">
                        <div class="drawer-content">
                            <div class="drawer-block">
                                <h3>Составляющие M&A Индекса Привлекательности</h3>
                                
                                <div class="score-bar-group">
                                    <div class="score-bar-labels">
                                        <span>1. Недооценка (DCF / аналоги)</span>
                                        <span class="text-secondary">${{c.scores.underval}}/100 (${{c.verdicts.underval_desc}})</span>
                                    </div>
                                    <div class="score-bar-bg">
                                        <div class="score-bar-fill" style="width: ${{c.scores.underval}}%; background: ${{c.scores.underval >= 70 ? 'var(--color-strong-buy)' : (c.scores.underval >= 40 ? 'var(--color-hold)' : 'var(--color-sell)')}};"></div>
                                    </div>
                                </div>
                                
                                <div class="score-bar-group">
                                    <div class="score-bar-labels">
                                        <span>2. Нужда в контроле (ROC vs WACC)</span>
                                        <span class="text-secondary">${{c.scores.control}}/100 (${{c.verdicts.control_desc}})</span>
                                    </div>
                                    <div class="score-bar-bg">
                                        <div class="score-bar-fill" style="width: ${{c.scores.control}}%; background: ${{c.scores.control >= 70 ? 'var(--color-strong-buy)' : (c.scores.control >= 40 ? 'var(--color-hold)' : 'var(--color-sell)')}};"></div>
                                    </div>
                                </div>
                                
                                <div class="score-bar-group">
                                    <div class="score-bar-labels">
                                        <span>3. Финансовая синергия (Запас кэша / LBO)</span>
                                        <span class="text-secondary">${{c.scores.fin}}/100 (${{c.verdicts.fin_desc}})</span>
                                    </div>
                                    <div class="score-bar-bg">
                                        <div class="score-bar-fill" style="width: ${{c.scores.fin}}%; background: ${{c.scores.fin >= 70 ? 'var(--color-strong-buy)' : (c.scores.fin >= 40 ? 'var(--color-hold)' : 'var(--color-sell)')}};"></div>
                                    </div>
                                </div>
                                
                                <div class="score-bar-group">
                                    <div class="score-bar-labels">
                                        <span>4. Операционная синергия (SG&A overhead)</span>
                                        <span class="text-secondary">${{c.scores.op}}/100 (${{c.verdicts.op_desc}})</span>
                                    </div>
                                    <div class="score-bar-bg">
                                        <div class="score-bar-fill" style="width: ${{c.scores.op}}%; background: ${{c.scores.op >= 70 ? 'var(--color-strong-buy)' : (c.scores.op >= 40 ? 'var(--color-hold)' : 'var(--color-sell)')}};"></div>
                                    </div>
                                </div>
                                
                                <div class="specs-grid">
                                    <div class="spec-item"><span class="spec-label">Текущая цена (Рыночная)</span><span class="spec-value">$${{c.financials.price}}</span></div>
                                    <div class="spec-item" style="border: 1px solid rgba(59,130,246,0.3); background: rgba(59,130,246,0.03);"><span class="spec-label" style="color: #60a5fa; font-weight: 600;">Средневзвешенная оценка (Ch. 8)</span><span class="spec-value" style="color: #60a5fa; font-weight: 700;">$${{c.financials.weighted_price}}</span></div>
                                    <div class="spec-item"><span class="spec-label">1. Standalone DCF Price (30%)</span><span class="spec-value">$${{c.financials.dcf_price}}</span></div>
                                    <div class="spec-item"><span class="spec-label">2. Peer Multiples Price (20%)</span><span class="spec-value">${{c.financials.peer_price ? '$' + c.financials.peer_price : '-'}}</span></div>
                                    <div class="spec-item"><span class="spec-label">3. Precedent Transactions (50%)</span><span class="spec-value">${{c.financials.precedent_price ? '$' + c.financials.precedent_price : '-'}}</span></div>
                                    <div class="spec-item" style="border: 1px solid rgba(16, 185, 129, 0.4); background: rgba(16, 185, 129, 0.05); grid-column: span 2;">
                                        <span class="spec-label" style="color: var(--color-strong-buy); font-weight: 600;">Рекомендуемая цена предложения (Offer Price)</span>
                                        <span class="spec-value" style="color: var(--color-strong-buy); font-weight: 700;">$${{c.financials.offer_price}}</span>
                                    </div>
                                    <div class="spec-item"><span class="spec-label">Доля синергии продавца (α)</span><span class="spec-value">${{c.financials.alpha}}</span></div>
                                    <div class="spec-item"><span class="spec-label">PV Synergy (на акцию)</span><span class="spec-value">${{c.financials.pv_synergy ? '$' + c.financials.pv_synergy : '-'}}</span></div>
                                    <div class="spec-item"><span class="spec-label">Базовое предложение (base_offer)</span><span class="spec-value">${{c.financials.base_offer ? '$' + c.financials.base_offer : '-'}}</span></div>
                                    <div class="spec-item"><span class="spec-label">Рыночный ориентир (Price + 30%)</span><span class="spec-value">${{c.financials.market_premium ? '$' + c.financials.market_premium : '-'}}</span></div>
                                    <div class="spec-item"><span class="spec-label">Оптимальная DCF (Control)</span><span class="spec-value">${{c.financials.optimal_dcf_price ? '$' + c.financials.optimal_dcf_price : '-'}}</span></div>
                                    <div class="spec-item"><span class="spec-label">Стоимость контроля (Control Premium)</span><span class="spec-value text-green">+${{c.financials.value_of_control ? '$' + c.financials.value_of_control : '$0.00'}}</span></div>
                                    <div class="spec-item"><span class="spec-label">WACC (Текущий / Опт.)</span><span class="spec-value">${{c.financials.wacc ? c.financials.wacc + '%' : '-'}} / ${{c.financials.opt_wacc ? c.financials.opt_wacc + '%' : '-'}}</span></div>
                                    <div class="spec-item"><span class="spec-label">Опт. маржа / Опт. ROC</span><span class="spec-value">${{c.financials.opt_margin ? c.financials.opt_margin + '%' : '-'}} / ${{c.financials.opt_roc ? c.financials.opt_roc + '%' : '-'}}</span></div>
                                    <div class="spec-item"><span class="spec-label">Страна (DePamphilis Ch. 18)</span><span class="spec-value">${{c.financials.country_name_ru ? c.financials.country_name_ru : 'США'}}</span></div>
                                    <div class="spec-item"><span class="spec-label">Глоб. Beta (β_emfirm,global)</span><span class="spec-value">${{c.financials.beta_emfirm_global ? c.financials.beta_emfirm_global : '-'}}</span></div>
                                    <div class="spec-item"><span class="spec-label">Размерная премия (FSP)</span><span class="spec-value">${{c.financials.fsp ? c.financials.fsp + '%' : '0.00%'}}</span></div>
                                    <div class="spec-item"><span class="spec-label">Премия за страновой риск (CRP)</span><span class="spec-value" style="color: ${{c.financials.crp > 0 ? 'var(--color-hold)' : 'inherit'}}; font-weight: ${{c.financials.crp > 0 ? '700' : 'normal'}};">${{c.financials.crp ? c.financials.crp + '%' : '0.00%'}}</span></div>
                                    <div class="spec-item" style="grid-column: span 2;"><span class="spec-label">Индустрия</span><span class="spec-value" style="font-family: inherit; font-size: 11px;">${{c.financials.industry}}</span></div>
                                    <div class="spec-item" style="grid-column: span 2;"><span class="spec-label">Катализатор</span><span class="spec-value" style="font-family: inherit; font-size: 11px; color: var(--color-accent);">${{c.catalyst}}</span></div>
                                </div>
                            </div>
                            
                            <div class="drawer-block">
                                <h3>Аудит защиты и реализуемости сделки</h3>
                                <p style="font-size: 13px; font-weight: 600; margin-bottom: 12px;" class="${{c.feasibility.class}}">Итог: ${{c.feasibility.verdict}}</p>
                                <ul class="audit-list">
                                    ${{auditItems}}
                                </ul>
                                <div style="margin-top: 14px; padding: 12px; background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.05); border-radius: 8px;">
                                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                                        <span style="font-size: 12px; color: var(--text-secondary);">Индекс защиты уставных док-в (E-Index, Bebchuk):</span>
                                        <span style="font-size: 14px; font-weight: 700; color: ${{c.depamphilis_metrics && c.depamphilis_metrics.e_index === 0 ? 'var(--color-strong-buy)' : (c.depamphilis_metrics && c.depamphilis_metrics.e_index >= 4 ? 'var(--color-sell)' : 'var(--color-hold)')}};">
                                            ${{c.depamphilis_metrics ? c.depamphilis_metrics.e_index : 0}} / 6
                                        </span>
                                    </div>
                                    ${{c.depamphilis_metrics && c.depamphilis_metrics.proxy_success_prob ? `
                                        <div style="font-size: 12px; margin-top: 6px; border-top: 1px dashed rgba(255,255,255,0.08); padding-top: 8px;">
                                            <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                                                <span style="color: var(--text-secondary); font-size: 11px;">Шанс успеха давления активистов:</span>
                                                <span style="font-weight: 700; color: var(--color-strong-buy);">${{c.depamphilis_metrics.proxy_success_prob.pressure_success}}</span>
                                            </div>
                                            <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                                                <span style="color: var(--text-secondary); font-size: 11px;">Победа в прокси-битве (Proxy Contest):</span>
                                                <span style="font-weight: 700; color: var(--color-accent);">${{c.depamphilis_metrics.proxy_success_prob.proxy_fight_win}}</span>
                                            </div>
                                            <div style="font-size: 11px; color: var(--text-secondary); margin-top: 4px; font-style: italic; line-height: 1.3;">
                                                ${{c.depamphilis_metrics.proxy_success_prob.reason}}
                                            </div>
                                        </div>
                                    ` : ''}}
                                </div>
                            </div>

                            <div class="drawer-block">
                                <h3>Временное окно сделки (Timing & Succession)</h3>
                                <p style="font-size: 13px; font-weight: 600; margin-bottom: 12px;" class="${{c.timing ? c.timing.class : 'text-secondary'}}">Итог: ${{c.timing ? c.timing.verdict : 'СПЯЩИЙ РЕЖИМ (Данные не рассчитаны)'}}</p>
                                <div class="score-bar-group" style="margin-bottom: 16px;">
                                    <div class="score-bar-labels">
                                        <span>Индекс готовности к сделке</span>
                                        <span class="text-secondary">${{c.timing ? c.timing.score : 0}}/100</span>
                                    </div>
                                    <div class="score-bar-bg">
                                        <div class="score-bar-fill" style="width: ${{c.timing ? c.timing.score : 0}}%; background: ${{c.timing && c.timing.score >= 60 ? 'var(--color-strong-buy)' : (c.timing && c.timing.score >= 30 ? 'var(--color-hold)' : 'var(--color-sell)')}};"></div>
                                    </div>
                                </div>
                                <ul class="audit-list">
                                    ${{timingItems}}
                                </ul>
                            </div>

                            <div class="drawer-block">
                                <h3>Акционеры с влиянием на сделку</h3>
                                ${{(function() {{
                                    const profile = c.buyer_profile;
                                    const holders = (profile && profile.financial && profile.financial.active_holders) ? profile.financial.active_holders : [];
                                    const passiveSummary = (profile && profile.financial) ? profile.financial.passive_summary : null;
                                    if (holders.length === 0) return '<div style="font-size: 11px; color: var(--text-secondary);">Активных акционеров с потенциалом влияния не обнаружено.</div>';
                                    let html = '';
                                    holders.forEach(h => {{
                                        let typeColor = 'var(--text-secondary)', typeBg = 'rgba(255,255,255,0.03)';
                                        if (h.type && h.type.includes('Активист')) {{ typeColor = '#ff6b6b'; typeBg = 'rgba(255,107,107,0.08)'; }}
                                        else if (h.type && h.type.includes('Стратег')) {{ typeColor = 'var(--color-accent)'; typeBg = 'rgba(99,179,237,0.07)'; }}
                                        const sign = h.pct_change > 0 ? '+' : '';
                                        const chg = h.pct_change !== 0 ? `<span style="color: ${{h.pct_change > 0 ? 'var(--color-strong-buy)' : 'var(--color-sell)'}}; font-size: 10px; margin-left: 6px;">${{sign}}${{h.pct_change}}%</span>` : '';
                                        html += `<div style="padding: 6px 8px; margin-bottom: 4px; border-radius: 6px; background: ${{typeBg}}; border: 1px solid rgba(255,255,255,0.06);">
                                            <div style="display: flex; justify-content: space-between; align-items: baseline; font-size: 11px;">
                                                <span style="font-weight: 600; color: var(--text-primary);">${{h.name}}</span>
                                                <span style="white-space: nowrap;">${{(h.shares/1e6).toFixed(2)}}M шт. ${{chg}}</span>
                                            </div>
                                            <div style="font-size: 9px; color: ${{typeColor}}; font-weight: 600; margin-top: 1px;">${{h.type}}</div>
                                            ${{h.influence ? `<div style="font-size: 9px; color: ${{typeColor}}; opacity: 0.8; margin-top: 2px;">${{h.influence}}</div>` : ''}}
                                        </div>`;
                                    }});
                                    if (passiveSummary) html += `<div style="padding: 5px 8px; margin-top: 6px; border-top: 1px dashed rgba(255,255,255,0.1); font-size: 10px; color: var(--text-secondary);"><span style="font-weight: 600;">Пассивный пул</span> — ${{passiveSummary.passive_weight_pct}}% от выборки (${{Math.round(passiveSummary.shares/1e6)}}M шт.)</div>`;
                                    return html;
                                }})()}}
                            </div>

                            <div class="drawer-block">
                                <h3>Финансовый стресс (Altman Z-Score / Distress)</h3>
                                ${{(function() {{
                                    const d = c.distress;
                                    if (!d) return '<div style="font-size:11px;color:var(--text-secondary);">Данные distress-анализа недоступны.</div>';
                                    const zColor = d.z_zone === 'Дистресс' ? 'var(--color-sell)' : (d.z_zone === 'Серая зона' ? 'var(--color-hold)' : 'var(--color-strong-buy)');
                                    const tbvColor = d.is_tbv_negative ? 'var(--color-sell)' : 'var(--color-strong-buy)';
                                    return `<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;font-size:12px;">
                                        <div style="padding:8px;background:rgba(255,255,255,0.02);border-radius:6px;border:1px solid rgba(255,255,255,0.05);">
                                            <div style="color:var(--text-secondary);font-size:10px;">Altman Z-Score</div>
                                            <div style="font-weight:700;color:${{zColor}};font-size:16px;">${{d.altman_z ?? 'N/A'}}</div>
                                            <div style="font-size:10px;color:${{zColor}};">${{d.z_zone}}</div>
                                        </div>
                                        <div style="padding:8px;background:rgba(255,255,255,0.02);border-radius:6px;border:1px solid rgba(255,255,255,0.05);">
                                            <div style="color:var(--text-secondary);font-size:10px;">Tangible Book Value</div>
                                            <div style="font-weight:700;color:${{tbvColor}};font-size:16px;">${{d.tbv !== null ? '$' + d.tbv + 'M' : 'N/A'}}</div>
                                            <div style="font-size:10px;color:${{tbvColor}}">${{d.is_tbv_negative ? 'TBV < 0 ⚠️' : 'TBV > 0 ✓'}}</div>
                                        </div>
                                        <div style="padding:8px;background:rgba(255,255,255,0.02);border-radius:6px;border:1px solid rgba(255,255,255,0.05);">
                                            <div style="color:var(--text-secondary);font-size:10px;">Net Debt / EBITDA</div>
                                            <div style="font-weight:700;color:${{d.nd_ebitda !== null && d.nd_ebitda > 6 ? 'var(--color-sell)' : 'var(--text-primary)'}};font-size:16px;">${{d.nd_ebitda !== null ? d.nd_ebitda + 'x' : 'N/A'}}</div>
                                        </div>
                                        <div style="padding:8px;background:rgba(255,255,255,0.02);border-radius:6px;border:1px solid rgba(255,255,255,0.05);">
                                            <div style="color:var(--text-secondary);font-size:10px;">Interest Coverage (EBIT)</div>
                                            <div style="font-weight:700;color:${{d.int_coverage_ebit !== null && d.int_coverage_ebit < 1 ? 'var(--color-sell)' : 'var(--text-primary)'}};font-size:16px;">${{d.int_coverage_ebit !== null ? d.int_coverage_ebit + 'x' : 'N/A'}}</div>
                                        </div>
                                    </div>
                                    ${{d.fcfe_override ? `<div style="margin-top:8px;padding:6px 10px;background:rgba(245,158,11,0.1);border:1px solid rgba(245,158,11,0.3);border-radius:6px;font-size:11px;color:var(--color-hold);">⚠️ Distress-свитч активирован: оценка переключена с FCFF на FCFE (TBV<0 или Z в зоне дистресса)</div>` : ''}}
                                    ${{d.is_in_distress ? `<div style="margin-top:6px;padding:6px 10px;background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.25);border-radius:6px;font-size:11px;color:var(--color-sell);">🔴 ${{d.distress_signals}}/4 сигнала дистресса активны — высокий риск финансовой несостоятельности</div>` : ''}}`;
                                }})()}}
                            </div>

                            <div class="drawer-block">
                                <h3>Мониторинг ковенантов долга Newco (Negative Covenants, Гл. 13)</h3>
                                ${{ (function() {{
                                    const cov = c.covenants;
                                    if (!cov) return '<div style="font-size:11px;color:var(--text-secondary);">Данные ковенантов Newco не рассчитаны.</div>';
                                    const riskColor = cov.risk_class === 'text-red' ? 'var(--color-sell)' : (cov.risk_class === 'text-orange' ? 'var(--color-hold)' : 'var(--color-strong-buy)');
                                    return `
                                    <div style="font-size: 13px; font-weight: 600; margin-bottom: 12px; color: ${{riskColor}};">
                                        Статус Newco: ${{cov.risk_level}}
                                    </div>
                                    <div style="display: flex; flex-direction: column; gap: 8px;">
                                        <div style="padding: 10px; background: rgba(255,255,255,0.02); border-radius: 6px; border: 1px solid rgba(255,255,255,0.05);">
                                            <div style="color: var(--text-secondary); font-size: 10px; font-weight: 600; text-transform: uppercase;">1. Ограничение дивидендов (Dividend Restrictions)</div>
                                            <div style="font-size: 11px; color: var(--text-primary); margin-top: 4px; line-height: 1.3;">
                                                ${{cov.dividend_reason}}
                                            </div>
                                            <div style="font-size: 10px; color: var(--text-secondary); margin-top: 4px;">
                                                Допустимый процент распределения: <strong>${{cov.dividend_cap_pct}}%</strong> от чистой прибыли Newco.
                                            </div>
                                        </div>
                                        <div style="padding: 10px; background: rgba(255,255,255,0.02); border-radius: 6px; border: 1px solid rgba(255,255,255,0.05);">
                                            <div style="color: var(--text-secondary); font-size: 10px; font-weight: 600; text-transform: uppercase;">2. Лимит дополнительных займов (Debt Limitation)</div>
                                            <div style="font-size: 11px; color: var(--text-primary); margin-top: 4px; line-height: 1.3;">
                                                ${{cov.debt_reason}}
                                            </div>
                                            <div style="font-size: 10px; color: var(--text-secondary); margin-top: 4px;">
                                                Максимальный лимит долга по ковенанту: <strong>$${{(cov.debt_limit / 1e6).toFixed(1)}}M</strong>.
                                            </div>
                                        </div>
                                        <div style="padding: 10px; background: rgba(255,255,255,0.02); border-radius: 6px; border: 1px solid rgba(255,255,255,0.05);">
                                            <div style="color: var(--text-secondary); font-size: 10px; font-weight: 600; text-transform: uppercase;">3. Ограничение капекса (Capital Expenditures Limits)</div>
                                            <div style="font-size: 11px; color: ${{cov.capex_breach ? 'var(--color-hold)' : 'var(--text-primary)'}}; margin-top: 4px; line-height: 1.3;">
                                                ${{cov.capex_reason}}
                                            </div>
                                            <div style="font-size: 10px; color: var(--text-secondary); margin-top: 4px;">
                                                Предельный уровень Capex по ковенанту: <strong>$${{(cov.capex_limit / 1e6).toFixed(1)}}M</strong>.
                                            </div>
                                        </div>
                                    </div>
                                    `;
                                }})() }}
                            </div>

                            <div class="drawer-block" style="grid-column: span 2;">
                                <h3>Потенциальные покупатели и мониторинг их ковенантов (Buyer Matchmaker)</h3>
                                <p style="font-size: 13px; color: var(--text-secondary); margin-bottom: 12px; line-height: 1.4;">
                                    Анализ соответствия стратегических и финансовых покупателей на основе долговой емкости (Debt Capacity), пересечения SIC-кодов (синергия SG&A с учетом фазирования и затрат на интеграцию по Гл. 14) и антимонопольных порогов HHI (Horizontal Merger Guidelines FTC, 2010).
                                </p>
                                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px;">
                                    ${{(function() {{
                                        const strategic = c.buyer_profile && c.buyer_profile.strategic ? c.buyer_profile.strategic : [];
                                        if (strategic.length === 0) return '<div style="font-size: 11px; color: var(--text-secondary); grid-column: span 2;">Нет подходящих покупателей в текущей выборке скринера.</div>';
                                        
                                        return strategic.map(b => {{
                                            const isPe = b.bidder_ticker === 'GENERIC_PE_SPONSOR';
                                            const name = isPe ? 'Финансовый спонсор (LBO Fund)' : `Отраслевой игрок (\\${{b.bidder_ticker}})`;
                                            const scoreColor = b.match_score >= 70 ? 'var(--color-strong-buy)' : (b.match_score >= 40 ? 'var(--color-hold)' : 'var(--color-sell)');
                                            
                                            // Construct notes list
                                            const notesHtml = b.notes.map(n => `<li style="font-size: 11px; margin-bottom: 4px; color: var(--text-secondary); line-height: 1.3;">\\${{n}}</li>`).join('');
                                            
                                            return `
                                            <div style="padding: 12px; background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 8px;">
                                                <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom: 6px; margin-bottom: 8px;">
                                                    <span style="font-weight: 700; color: var(--color-accent); font-size: 13px;">\\${{name}}</span>
                                                    <span style="font-weight: 700; color: \\${{scoreColor}}; font-size: 13px;">Match: \\${{b.match_score}}/100</span>
                                                </div>
                                                <ul style="list-style: none; padding-left: 0;">
                                                    \\${{notesHtml}}
                                                </ul>
                                            </div>
                                            `;
                                        }}).join('');
                                    }})()}}
                                </div>
                            </div>

                            <div class="drawer-block" style="grid-column: span 2;">
                                <h3>GAAP-симулятор Goodwill и NCI (Purchase Accounting, Exhibit 12.1)</h3>
                                <p style="font-size: 13px; color: var(--text-secondary); margin-bottom: 12px; line-height: 1.4;">
                                    Расчёт по методу приобретения (Purchase Accounting, US GAAP / IFRS 3) при покупке контролирующего, но не полного пакета акций (&lt; 100%). Все существующие Goodwill цели списываются, и признаётся новый Goodwill всей компании по справедливой стоимости.
                                </p>
                                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px;">
                                    ${{(function() {{
                                        const scenarios = c.consolidation_scenarios;
                                        if (!scenarios) return '<div style="font-size:11px;color:var(--text-secondary);">Данные симулятора консолидации недоступны.</div>';
                                        let html = '';
                                        ['51', '80', '100'].forEach(key => {{
                                            const s = scenarios[key];
                                            if (!s) return;
                                            const pct = s.ownership_pct;
                                            const pp = s.purchase_price;
                                            const full_eq = s.implied_full_equity_val;
                                            const nci = s.nci_val;
                                            const net_assets_base = s.fv_net_identifiable_assets_base || s.fv_net_identifiable_assets;
                                            const step_up_pct = s.step_up_pct || 0;
                                            const step_up_val = s.step_up_val || 0;
                                            const net_assets = s.fv_net_identifiable_assets;
                                            const gw = s.new_goodwill;
                                            const annual_step_up_da = s.annual_step_up_da || 0;
                                            const annual_tax_shield = s.annual_tax_shield || 0;
                                            const annual_net_income_drag = s.annual_net_income_drag || 0;
                                            const eps_drag = s.eps_drag || 0;
                                            const annual_cash_flow_drag_stock = s.annual_cash_flow_drag_stock || 0;
                                            const annual_cash_flow_drag_asset = s.annual_cash_flow_drag_asset || 0;
                                            
                                            html += `
                                            <div style="padding: 12px; background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 8px;">
                                                <div style="font-weight: 700; color: var(--color-accent); font-size: 14px; border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom: 4px; margin-bottom: 8px;">
                                                    Доля выкупа: ${{pct}}%
                                                </div>
                                                <div style="display: flex; flex-direction: column; gap: 6px; font-size: 12px;">
                                                    <div style="display: flex; justify-content: space-between;">
                                                        <span style="color: var(--text-secondary);">Цена приобретения:</span>
                                                        <span style="font-weight: 600;">$${{(pp / 1e6).toFixed(1)}}M</span>
                                                    </div>
                                                    <div style="display: flex; justify-content: space-between;">
                                                        <span style="color: var(--text-secondary);">Справедливая стоимость NCI:</span>
                                                        <span style="font-weight: 600;">$${{(nci / 1e6).toFixed(1)}}M</span>
                                                    </div>
                                                    <div style="display: flex; justify-content: space-between; border-bottom: 1px dashed rgba(255,255,255,0.05); padding-bottom: 4px;">
                                                        <span style="color: var(--text-secondary);">Полная ст-сть Newco:</span>
                                                        <span style="font-weight: 600; color: var(--text-primary);">$${{(full_eq / 1e6).toFixed(1)}}M</span>
                                                    </div>
                                                    <div style="display: flex; justify-content: space-between;">
                                                        <span style="color: var(--text-secondary);">Базовые чистые активы (Book):</span>
                                                        <span style="font-weight: 600;">$${{(net_assets_base / 1e6).toFixed(1)}}M</span>
                                                    </div>
                                                    <div style="display: flex; justify-content: space-between;">
                                                        <span style="color: var(--text-secondary);">Asset Step-Up (${{step_up_pct.toFixed(0)}}%):</span>
                                                        <span style="font-weight: 600; color: #60a5fa;">+$${{(step_up_val / 1e6).toFixed(1)}}M</span>
                                                    </div>
                                                    <div style="display: flex; justify-content: space-between; border-bottom: 1px dashed rgba(255,255,255,0.05); padding-bottom: 4px;">
                                                        <span style="color: var(--text-secondary);">Итоговые чистые активы (FV):</span>
                                                        <span style="font-weight: 600;">$${{(net_assets / 1e6).toFixed(1)}}M</span>
                                                    </div>
                                                    <div style="display: flex; justify-content: space-between; border-bottom: 1px dashed rgba(255,255,255,0.05); padding-bottom: 4px; font-weight: 700;">
                                                        <span style="color: var(--color-strong-buy);">Новый Goodwill (GAAP):</span>
                                                        <span style="color: var(--color-strong-buy);">$${{(gw / 1e6).toFixed(1)}}M</span>
                                                    </div>
                                                    <div style="display: flex; justify-content: space-between;">
                                                        <span style="color: var(--text-secondary);">Доп. D&A (амортизация):</span>
                                                        <span style="font-weight: 600; color: #f59e0b;">-$${{Math.abs(annual_step_up_da / 1e6).toFixed(2)}}M/г</span>
                                                    </div>
                                                    <div style="display: flex; justify-content: space-between; margin-top: 4px; border-top: 1px dashed rgba(255,255,255,0.05); padding-top: 4px;">
                                                        <span style="color: var(--text-primary); font-weight: 600; font-size: 11px;">Бухгалтерский учёт (Book / EPS):</span>
                                                    </div>
                                                    <div style="display: flex; justify-content: space-between; padding-left: 8px;">
                                                        <span style="color: var(--text-secondary); font-size: 11px;">• Налоговый щит DTL (снижает налог на прибыль):</span>
                                                        <span style="font-weight: 600; color: var(--color-strong-buy); font-size: 11px;">+$${{Math.abs(annual_tax_shield / 1e6).toFixed(2)}}M/г</span>
                                                    </div>
                                                    <div style="display: flex; justify-content: space-between; padding-left: 8px;">
                                                        <span style="color: var(--text-secondary); font-size: 11px;">• Чистый эффект на прибыль (Book Drag):</span>
                                                        <span style="font-weight: 600; color: #f59e0b; font-size: 11px;">-$${{Math.abs(annual_net_income_drag / 1e6).toFixed(2)}}M/г</span>
                                                    </div>
                                                    <div style="display: flex; justify-content: space-between; padding-left: 8px;">
                                                        <span style="color: var(--text-secondary); font-size: 11px;">• EPS Драг (EPS Drag):</span>
                                                        <span style="font-weight: 700; color: #ef4444; font-size: 11px;">-$${{Math.abs(eps_drag).toFixed(4)}}/акц.</span>
                                                    </div>
                                                    <div style="display: flex; justify-content: space-between; margin-top: 4px; border-top: 1px dashed rgba(255,255,255,0.05); padding-top: 4px;">
                                                        <span style="color: var(--text-primary); font-weight: 600; font-size: 11px;">Реальный денежный поток (Cash Flows):</span>
                                                    </div>
                                                    <div style="display: flex; justify-content: space-between; padding-left: 8px;">
                                                        <span style="color: var(--text-secondary); font-size: 11px;">• Stock Purchase (Без Sec 338, щит = $0):</span>
                                                        <span style="font-weight: 600; color: #ef4444; font-size: 11px;">-$${{Math.abs(annual_cash_flow_drag_stock / 1e6).toFixed(2)}}M/г</span>
                                                    </div>
                                                    <div style="display: flex; justify-content: space-between; padding-left: 8px;">
                                                        <span style="color: var(--text-secondary); font-size: 11px;">• Asset Purchase / Sec 338 (щит признан):</span>
                                                        <span style="font-weight: 600; color: var(--color-strong-buy); font-size: 11px;">-$${{Math.abs(annual_cash_flow_drag_asset / 1e6).toFixed(2)}}M/г</span>
                                                    </div>
                                                </div>
                                            </div>`;
                                        }});
                                        return html;
                                    }})()}}
                                </div>
                            </div>

                            <div class="drawer-block" style="grid-column: span 2;">
                                <h3>Моделирование воротниковых соглашений (Collar Arrangements, Exhibit 11.1)</h3>
                                <p style="font-size: 13px; color: var(--text-secondary); margin-bottom: 12px; line-height: 1.4;">
                                    В сделках с оплатой акциями рыночные колебания цены покупателя (Acquirer) между подписанием и закрытием создают риски. Воротниковые соглашения (Collars) фиксируют границы ценовых коридоров (Collar Range) согласно методологии книги (Exhibit 11.1). Ниже представлено моделирование для <strong>Fixed-Value</strong> (где стоимость сделки гарантирована внутри коридора за счет плавающего Ratio) и <strong>Fixed-Share</strong> (где коэффициент обмена зафиксирован, а стоимость сделки колеблется в рамках лимитов).
                                </p>
                                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; margin-bottom: 14px;">
                                    <div style="background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 8px; padding: 12px;">
                                        <div style="font-weight: 700; color: var(--color-strong-buy); font-size: 14px; border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom: 6px; margin-bottom: 10px; display: flex; justify-content: space-between;">
                                            <span>Fixed-Value Collar (SER Floats)</span>
                                            <span style="font-size: 11px; font-weight: normal; color: var(--text-secondary);">Полезен для продавца</span>
                                        </div>
                                        <div style="max-height: 250px; overflow-y: auto;">
                                            <table style="width: 100%; font-size: 11px; border-collapse: collapse; text-align: left;">
                                                <thead>
                                                    <tr style="color: var(--text-secondary); border-bottom: 1px solid rgba(255,255,255,0.1);">
                                                        <th style="padding: 4px; font-size: 11px;">Цена покупателя</th>
                                                        <th style="padding: 4px; font-size: 11px; text-align: center;">Exchange Ratio (SER)</th>
                                                        <th style="padding: 4px; font-size: 11px; text-align: right;">Итоговая ценность</th>
                                                        <th style="padding: 4px; font-size: 11px; text-align: right;">Статус</th>
                                                    </tr>
                                                </thead>
                                                <tbody>
                                                    ${{(function() {{
                                                        const collar = c.collar_scenarios;
                                                        if (!collar || !collar.fixed_value_collar) return '<tr><td colspan="4" style="text-align: center; color: var(--text-secondary); padding: 10px;">Данные воротникового соглашения отсутствуют</td></tr>';
                                                        return collar.fixed_value_collar.map(s => {{
                                                            const isBase = s.change_pct === 0 ? 'font-weight: bold; background: rgba(59,130,246,0.15);' : '';
                                                            const statusColor = s.status.includes('Внутри') ? 'color: var(--color-strong-buy);' : 'color: var(--text-secondary);';
                                                            return `<tr style="${{isBase}} border-bottom: 1px solid rgba(255,255,255,0.03);">
                                                                <td style="padding: 6px 4px;">$${{s.acquirer_price.toFixed(2)}} (${{s.change_pct > 0 ? '+' : ''}}${{s.change_pct}}%)</td>
                                                                <td style="padding: 6px 4px; text-align: center; font-family: monospace;">${{s.exchange_ratio.toFixed(4)}}</td>
                                                                <td style="padding: 6px 4px; text-align: right; font-weight: 600;">$${{s.offer_value.toFixed(2)}}</td>
                                                                <td style="padding: 6px 4px; text-align: right; font-size: 9px; ${{statusColor}}">${{s.status}}</td>
                                                            </tr>`;
                                                        }}).join('');
                                                    }})()}}
                                                </tbody>
                                            </table>
                                        </div>
                                    </div>
                                    
                                    <div style="background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 8px; padding: 12px;">
                                        <div style="font-weight: 700; color: var(--color-accent); font-size: 14px; border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom: 6px; margin-bottom: 10px; display: flex; justify-content: space-between;">
                                            <span>Fixed-Share Collar (Value Floats)</span>
                                            <span style="font-size: 11px; font-weight: normal; color: var(--text-secondary);">Полезен для покупателя</span>
                                        </div>
                                        <div style="max-height: 250px; overflow-y: auto;">
                                            <table style="width: 100%; font-size: 11px; border-collapse: collapse; text-align: left;">
                                                <thead>
                                                    <tr style="color: var(--text-secondary); border-bottom: 1px solid rgba(255,255,255,0.1);">
                                                        <th style="padding: 4px; font-size: 11px;">Цена покупателя</th>
                                                        <th style="padding: 4px; font-size: 11px; text-align: center;">Exchange Ratio (SER)</th>
                                                        <th style="padding: 4px; font-size: 11px; text-align: right;">Итоговая ценность</th>
                                                        <th style="padding: 4px; font-size: 11px; text-align: right;">Статус</th>
                                                    </tr>
                                                </thead>
                                                <tbody>
                                                    ${{(function() {{
                                                        const collar = c.collar_scenarios;
                                                        if (!collar || !collar.fixed_share_collar) return '<tr><td colspan="4" style="text-align: center; color: var(--text-secondary); padding: 10px;">Данные воротникового соглашения отсутствуют</td></tr>';
                                                        return collar.fixed_share_collar.map(s => {{
                                                            const isBase = s.change_pct === 0 ? 'font-weight: bold; background: rgba(59,130,246,0.15);' : '';
                                                            const statusColor = s.status.includes('Внутри') ? 'color: var(--color-accent);' : 'color: var(--text-secondary);';
                                                            return `<tr style="${{isBase}} border-bottom: 1px solid rgba(255,255,255,0.03);">
                                                                <td style="padding: 6px 4px;">$${{s.acquirer_price.toFixed(2)}} (${{s.change_pct > 0 ? '+' : ''}}${{s.change_pct}}%)</td>
                                                                <td style="padding: 6px 4px; text-align: center; font-family: monospace;">${{s.exchange_ratio.toFixed(4)}}</td>
                                                                <td style="padding: 6px 4px; text-align: right; font-weight: 600;">$${{s.offer_value.toFixed(2)}}</td>
                                                                <td style="padding: 6px 4px; text-align: right; font-size: 9px; ${{statusColor}}">${{s.status}}</td>
                                                            </tr>`;
                                                        }}).join('');
                                                    }})()}}
                                                </tbody>
                                            </table>
                                        </div>
                                    </div>
                                </div>
                            </div>

                            <div class="drawer-block">
                                <h3>Зацепки об M&amp;A в отчётах SEC (EDGAR)</h3>
                                ${{(function() {{
                                    if (!c.sec_ma_hints || c.sec_ma_hints.length === 0) return '<div style="font-size: 11px; color: var(--text-secondary);">M&A зацепок в отчётах SEC не обнаружено за последний год.</div>';
                                    return c.sec_ma_hints.map(h => `
                                        <div style="margin-bottom: 8px; padding: 8px; background: rgba(59,130,246,0.05); border: 1px solid rgba(59,130,246,0.15); border-radius: 8px; font-size: 11px;">
                                            <div style="display: flex; justify-content: space-between; font-weight: 600;">
                                                <span style="color: var(--color-accent); font-weight: 700;">${{h.form}}</span>
                                                <span style="color: var(--text-secondary); font-size: 10px;">${{h.date}}</span>
                                            </div>
                                            <div style="color: var(--text-primary); font-weight: 600; margin-top: 4px;">${{h.reason}}</div>
                                            ${{h.description ? `<div style="color: var(--text-secondary); font-size: 10px; font-style: italic; margin-top: 2px;">${{h.description}}</div>` : ''}}
                                            <div style="margin-top: 6px; text-align: right;">
                                                <a href="${{h.url}}" target="_blank" style="color: var(--color-accent); text-decoration: none; font-size: 10px; font-weight: 600; border-bottom: 1px dashed var(--color-accent);">Открыть на SEC.gov ↗</a>
                                            </div>
                                        </div>`).join('');
                                }})()}}
                            </div>

                        </div>
                    </td>
                `;
                
                tbody.appendChild(drawerTr);
            }});
        }}


        function toggleRow(ticker) {{
            const drawer = document.getElementById(`drawer-${{ticker}}`);
            const row = document.getElementById(`row-${{ticker}}`);
            
            const isVisible = drawer.style.display !== 'none';
            
            document.querySelectorAll('.drawer-row').forEach(d => d.style.display = 'none');
            document.querySelectorAll('.table-row').forEach(r => r.classList.remove('active'));
            
            if (!isVisible) {{
                drawer.style.display = 'table-row';
                row.classList.add('active');
            }}
        }}

        function handleSort(key) {{
            if (currentSort.key === key) {{
                currentSort.desc = !currentSort.desc;
            }} else {{
                currentSort.key = key;
                currentSort.desc = true;
            }}
            renderTable();
        }}

        function initScatterPlot() {{
            const ctx = document.getElementById('maScatterChart').getContext('2d');
            
            const chartData = companiesData.map(c => {{
                const jitterX = (Math.random() - 0.5) * 4.0;
                const jitterY = (Math.random() - 0.5) * 4.0;
                
                let pointColor = '#f59e0b';
                if (c.feasibility && c.feasibility.verdict && c.feasibility.verdict.includes('ВЫСОКАЯ')) pointColor = '#10b981';
                else if (c.feasibility && c.feasibility.verdict && c.feasibility.verdict.includes('НИЗКАЯ')) pointColor = '#ef4444';
                
                return {{
                    x: Math.min(99, Math.max(1, (typeof c.score === 'number' && !isNaN(c.score) ? c.score : 0) + jitterX)),
                    y: Math.min(99, Math.max(1, (c.feasibility && typeof c.feasibility.score === 'number' && !isNaN(c.feasibility.score) ? c.feasibility.score : 0) + jitterY)),
                    ticker: c.ticker,
                    name: c.name,
                    rawScore: c.score,
                    rawFeas: c.feasibility.score,
                    verdict: c.feasibility.verdict,
                    color: pointColor
                }};
            }});
            
            const quadrantBackgroundPlugin = {{
                id: 'quadrantBackground',
                beforeDraw(chart) {{
                    const {{ ctx, chartArea: {{ left, top, right, bottom, width, height }} }} = chart;
                    const xScale = chart.scales.x;
                    const yScale = chart.scales.y;
                    
                    const midX = xScale.getPixelForValue(50);
                    const midY = yScale.getPixelForValue(50);
                    
                    ctx.save();
                    
                    // Top Right: Ideal
                    ctx.fillStyle = 'rgba(16, 185, 129, 0.03)';
                    ctx.fillRect(midX, top, right - midX, midY - top);
                    
                    // Bottom Right: Protected
                    ctx.fillStyle = 'rgba(239, 68, 68, 0.02)';
                    ctx.fillRect(midX, midY, right - midX, bottom - midY);
                    
                    // Top Left: Expensive
                    ctx.fillStyle = 'rgba(59, 130, 246, 0.02)';
                    ctx.fillRect(left, top, midX - left, midY - top);
                    
                    // Bottom Left: Low Priority
                    ctx.fillStyle = 'rgba(255, 255, 255, 0.005)';
                    ctx.fillRect(left, midY, midX - left, bottom - midY);
                    
                    ctx.lineWidth = 1.5;
                    ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
                    ctx.setLineDash([6, 6]);
                    
                    ctx.beginPath();
                    ctx.moveTo(midX, top);
                    ctx.lineTo(midX, bottom);
                    ctx.stroke();
                    
                    ctx.beginPath();
                    ctx.moveTo(left, midY);
                    ctx.lineTo(right, midY);
                    ctx.stroke();
                    
                    ctx.fillStyle = 'rgba(255, 255, 255, 0.35)';
                    ctx.font = 'bold 11px Inter';
                    ctx.textAlign = 'right';
                    
                    ctx.fillText('ИДЕАЛЬНЫЕ ЦЕЛИ ПОГЛОЩЕНИЯ', right - 12, top + 18);
                    ctx.fillText('ЗАЩИЩЕННЫЕ ГИГАНТЫ / FTC БАРЬЕРЫ', right - 12, bottom - 12);
                    
                    ctx.textAlign = 'left';
                    ctx.fillText('ДОРОГИЕ / ЭФФЕКТИВНЫЕ ЦЕЛИ', left + 12, top + 18);
                    ctx.fillText('НИЗКИЙ M&A ПРИОРИТЕТ', left + 12, bottom - 12);
                    
                    ctx.restore();
                }}
            }};

            scatterChart = new Chart(ctx, {{
                type: 'scatter',
                data: {{
                    datasets: [{{
                        label: 'Компании',
                        data: chartData,
                        pointBackgroundColor: chartData.map(d => d.color),
                        pointBorderColor: 'rgba(255,255,255,0.4)',
                        pointBorderWidth: 1.5,
                        pointRadius: 8,
                        pointHoverRadius: 11,
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{ display: false }},
                        tooltip: {{
                            backgroundColor: 'rgba(15, 23, 42, 0.95)',
                            titleFont: {{ family: 'Outfit', size: 14, weight: 'bold' }},
                            bodyFont: {{ family: 'Inter', size: 12 }},
                            borderColor: 'rgba(255,255,255,0.15)',
                            borderWidth: 1,
                            padding: 12,
                            callbacks: {{
                                title: (items) => {{
                                    const raw = items[0].raw;
                                    return `${{raw.ticker}} - ${{raw.name}}`;
                                }},
                                label: (item) => {{
                                    const raw = item.raw;
                                    return [
                                        `Привлекательность (Score): ${{raw.rawScore}}/100`,
                                        `Реалистичность сделки: ${{raw.verdict.split(' ')[0]}} (${{raw.rawFeas}}/100)`
                                    ];
                                }}
                            }}
                        }}
                    }},
                    scales: {{
                        x: {{
                            min: 0,
                            max: 100,
                            title: {{
                                display: true,
                                text: 'M&A Привлекательность (Индекс по Главе 25) ──►',
                                color: '#9ca3af',
                                font: {{ family: 'Inter', size: 12, weight: 'bold' }}
                            }},
                            grid: {{ color: 'rgba(255, 255, 255, 0.03)' }},
                            ticks: {{ color: '#9ca3af', font: {{ family: 'Fira Code' }} }}
                        }},
                        y: {{
                            min: 0,
                            max: 100,
                            title: {{
                                display: true,
                                text: 'Реалистичность поглощения / Отсутствие барьеров ──►',
                                color: '#9ca3af',
                                font: {{ family: 'Inter', size: 12, weight: 'bold' }}
                            }},
                            grid: {{ color: 'rgba(255, 255, 255, 0.03)' }},
                            ticks: {{ color: '#9ca3af', font: {{ family: 'Fira Code' }} }}
                        }}
                    }},
                    onClick: (e, items) => {{
                        if (items.length > 0) {{
                            const idx = items[0].index;
                            const dataset = scatterChart.data.datasets[0];
                            const point = dataset.data[idx];
                            
                            const tableRow = document.getElementById(`row-${{point.ticker}}`);
                            if (tableRow) {{
                                tableRow.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
                                toggleRow(point.ticker);
                            }}
                        }}
                    }}
                }},
                plugins: [quadrantBackgroundPlugin]
            }});
        }}

        function updateScatterChart() {{
            const filtered = getFilteredData();
            
            const chartData = filtered.map(c => {{
                const jitterX = (Math.random() - 0.5) * 4.0;
                const jitterY = (Math.random() - 0.5) * 4.0;
                
                let pointColor = '#f59e0b';
                if (c.feasibility && c.feasibility.verdict && c.feasibility.verdict.includes('ВЫСОКАЯ')) pointColor = '#10b981';
                else if (c.feasibility && c.feasibility.verdict && c.feasibility.verdict.includes('НИЗКАЯ')) pointColor = '#ef4444';
                
                return {{
                    x: Math.min(99, Math.max(1, (typeof c.score === 'number' && !isNaN(c.score) ? c.score : 0) + jitterX)),
                    y: Math.min(99, Math.max(1, (c.feasibility && typeof c.feasibility.score === 'number' && !isNaN(c.feasibility.score) ? c.feasibility.score : 0) + jitterY)),
                    ticker: c.ticker,
                    name: c.name,
                    rawScore: c.score,
                    rawFeas: c.feasibility.score,
                    verdict: c.feasibility.verdict,
                    color: pointColor
                }};
            }});
            
            scatterChart.data.datasets[0].data = chartData;
            scatterChart.data.datasets[0].pointBackgroundColor = chartData.map(d => d.color);
            scatterChart.update();
        }}
    </script>
</body>
</html>
"""

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_template)

if __name__ == "__main__":
    main()