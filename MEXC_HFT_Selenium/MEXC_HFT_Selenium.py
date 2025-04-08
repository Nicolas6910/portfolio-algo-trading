import time
import threading
import requests
import hmac
import hashlib
import json
import urllib.parse
import sys
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

API_KEY = "mx0vglRxsGOcXbZ0eZ"
API_SECRET = "059e76d13c204406a0736ecf120dade3"
BASE_URL = "https://contract.mexc.com"

def get_timestamp():
    return str(int(time.time() * 1000))

def get_request_param_string(params):
    if not params:
        return ""
    sorted_items = sorted(params.items())
    encoded_items = []
    for key, value in sorted_items:
        value = "" if value is None else str(value)
        encoded_value = urllib.parse.quote(value, safe='')
        encoded_items.append(f"{key}={encoded_value}")
    return "&".join(encoded_items)

def sign_request(method, endpoint, params, timestamp):
    if method.upper() in ["GET", "DELETE"]:
        param_string = get_request_param_string(params)
    elif method.upper() == "POST":
        param_string = json.dumps(params, separators=(',', ':'), ensure_ascii=False) if params else ""
    else:
        param_string = ""
    target_string = API_KEY + timestamp + param_string
    signature = hmac.new(API_SECRET.encode(), target_string.encode(), hashlib.sha256).hexdigest()
    return signature

def send_request(method, endpoint, params=None):
    if params is None:
        params = {}
    timestamp = get_timestamp()
    signature = sign_request(method, endpoint, params, timestamp)
    headers = {
        "Content-Type": "application/json",
        "ApiKey": API_KEY,
        "Request-Time": timestamp,
        "Signature": signature,
    }
    url = BASE_URL + endpoint
    try:
        if method.upper() == "GET":
            resp = requests.get(url, headers=headers, params=params, timeout=10)
        elif method.upper() == "POST":
            resp = requests.post(url, headers=headers, json=params, timeout=10)
        elif method.upper() == "DELETE":
            resp = requests.delete(url, headers=headers, params=params, timeout=10)
        else:
            raise ValueError("Méthode HTTP non supportée")
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"[API ERROR] {e}")
    return None

def get_doge_perp_price():
    url = "https://contract.mexc.com/api/v1/contract/ticker"
    params = {"symbol": "SHIB_USDT"}
    try:
        r = requests.get(url, params=params, timeout=5)
        data = r.json()
        if "data" in data and data["data"]:
            return float(data["data"]["lastPrice"])
    except Exception as e:
        print(f"[API] Erreur récupération prix SHIB perp: {e}")
    return None

def get_open_positions(symbol="SHIB_USDT"):
    endpoint = "/api/v1/private/position/open_positions"
    params = {"symbol": symbol}
    return send_request("GET", endpoint, params)

def get_open_orders(symbol="SHIB_USDT"):
    endpoint = "/api/v1/private/order/list/open_orders"
    params = {"symbol": symbol}
    return send_request("GET", endpoint, params)

UI_DELAY = 0.2
EXCHANGE_DELAY = 3.0

def scroll_top(driver):
    driver.execute_script("window.scrollTo(0, 0);")
    time.sleep(UI_DELAY)

def scroll_bottom(driver):
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(UI_DELAY)

chrome_options = Options()
chrome_options.add_argument(r"user-data-dir=C:\Users\nicol\AppData\Local\Google\Chrome\User Data")
chrome_options.add_argument("profile-directory=Default")
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=chrome_options)
wait = WebDriverWait(driver, 20)
driver.get("https://futures.mexc.co/fr-FR/exchange/SHIB_USDT?type=linear_swap")
time.sleep(5)

def try_click_confirmer_modal():
    confirm_buttons = driver.find_elements(By.XPATH, "//button[contains(@class,'ant-btn')]//span[text()='Confirmer']")
    for btn in confirm_buttons:
        try:
            if btn.is_displayed() and btn.is_enabled():
                btn.click()
                time.sleep(UI_DELAY)
                print("[MODAL] Bouton 'Confirmer' cliqué.")
        except:
            pass
    time.sleep(UI_DELAY)

def confirm_rappel_des_risques():
    try:
        risk_div = WebDriverWait(driver, 1).until(EC.visibility_of_element_located((By.XPATH, "//div[text()='Le prix de déclenchement est proche du prix actuel, il peut donc être déclenché immédiatement.']")))
        time.sleep(UI_DELAY)
        if risk_div:
            print("[POPUP] Fenêtre 'Rappel des risques' détectée.")
            confirm_btn = WebDriverWait(driver, 5).until(EC.element_to_be_clickable((By.XPATH, "//div[@class='ant-modal-content']//span[text()='Confirmer']")))
            time.sleep(UI_DELAY)
            confirm_btn.click()
            time.sleep(UI_DELAY)
            print("[POPUP] Confirmation popup Rappel des risques cliquée.")
    except:
        pass
    try_click_confirmer_modal()

def open_market_order(montant="0.1"):
    scroll_top(driver)
    try_click_confirmer_modal()
    print(f"[MARKET] Ouverture Market => {montant} USDT.")
    time.sleep(UI_DELAY)
    try:
        ouvrir_btn = wait.until(EC.element_to_be_clickable((By.XPATH, "//span[@data-testid='contract-trade-order-form-tab-open']")))
        time.sleep(UI_DELAY)
        ouvrir_btn.click()
        time.sleep(UI_DELAY)
    except Exception as e:
        print("[MARKET] Erreur clic 'Ouvrir':", e)
        try_click_confirmer_modal()
    try:
        market_tab = wait.until(EC.element_to_be_clickable((By.XPATH, "//span[text()='Market']")))
        time.sleep(UI_DELAY)
        market_tab.click()
        time.sleep(UI_DELAY)
    except Exception as e:
        print("[MARKET] Erreur clic 'Market':", e)
        try_click_confirmer_modal()
    try:
        inputs = driver.find_elements(By.XPATH, "//input[@autocomplete='off' and contains(@class,'ant-input') and @value='']")
        time.sleep(UI_DELAY)
        if inputs:
            inp = inputs[0]
            inp.click()
            time.sleep(UI_DELAY)
            inp.clear()
            time.sleep(UI_DELAY)
            inp.send_keys(str(montant))
            time.sleep(UI_DELAY)
        else:
            print("[MARKET] Aucun input pour le montant.")
    except Exception as e:
        print("[MARKET] Erreur saisie du montant:", e)
    scroll_bottom(driver)
    try_click_confirmer_modal()
    try:
        btn_long = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "button[data-testid='contract-trade-open-long-btn']")))
        time.sleep(UI_DELAY)
        btn_long.click()
        time.sleep(UI_DELAY)
        print(f"[MARKET] Ordre Market {montant} USDT envoyé.")
    except Exception as e:
        print("[MARKET] Erreur clic 'Ouvrir Long':", e)
        try_click_confirmer_modal()

def open_limit_order(montant, limit_price):
    scroll_top(driver)
    try_click_confirmer_modal()
    print(f"[LIMIT] Ouverture Limit => {montant} USDT @ {limit_price:.5f}")
    time.sleep(UI_DELAY)
    try:
        ouvrir_btn = wait.until(EC.element_to_be_clickable((By.XPATH, "//span[@data-testid='contract-trade-order-form-tab-open']")))
        time.sleep(UI_DELAY)
        ouvrir_btn.click()
        time.sleep(UI_DELAY)
    except Exception as e:
        print("[LIMIT] Erreur clic 'Ouvrir':", e)
        try_click_confirmer_modal()
    try:
        limit_tab = wait.until(EC.element_to_be_clickable((By.XPATH, "//span[contains(@class,'EntrustTabs_active__') and text()='Limit']")))
        time.sleep(UI_DELAY)
        limit_tab.click()
        time.sleep(UI_DELAY)
    except Exception as e:
        print("[LIMIT] Erreur clic 'Limit':", e)
        try_click_confirmer_modal()
    try:
        price_input = wait.until(EC.element_to_be_clickable((By.XPATH, "(//input[@autocomplete='off' and contains(@class,'ant-input')])[1]")))
        time.sleep(UI_DELAY)
        price_input.click()
        time.sleep(UI_DELAY)
        price_input.send_keys(Keys.CONTROL, "a")
        time.sleep(UI_DELAY)
        price_input.send_keys(Keys.BACKSPACE)
        time.sleep(UI_DELAY)
        price_input.send_keys(str(limit_price))
        time.sleep(UI_DELAY)
        print(f"[LIMIT] Saisi du prix: {limit_price:.5f}")
    except Exception as e:
        print("[LIMIT] Erreur saisie prix limit:", e)
    try:
        inputs = driver.find_elements(By.XPATH, "//input[@autocomplete='off' and contains(@class,'ant-input') and @value='']")
        time.sleep(UI_DELAY)
        if inputs:
            amt_inp = inputs[0]
            amt_inp.click()
            time.sleep(UI_DELAY)
            amt_inp.clear()
            time.sleep(UI_DELAY)
            amt_inp.send_keys(str(montant))
            time.sleep(UI_DELAY)
            print(f"[LIMIT] Montant = {montant} USDT")
        else:
            print("[LIMIT] Aucun input pour le montant.")
    except Exception as e:
        print("[LIMIT] Erreur saisie montant:", e)
    scroll_bottom(driver)
    try_click_confirmer_modal()
    try:
        btn_long = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "button[data-testid='contract-trade-open-long-btn']")))
        time.sleep(UI_DELAY)
        btn_long.click()
        time.sleep(UI_DELAY)
        print(f"[LIMIT] Ordre Limit envoyé: {montant} USDT @ {limit_price:.5f}")
    except Exception as e:
        print("[LIMIT] Erreur clic 'Ouvrir Long':", e)
        try_click_confirmer_modal()

def open_stop_loss(stop_price, is_long=True):
    scroll_top(driver)
    try_click_confirmer_modal()
    print(f"[SL] Ouverture d'un Stop-Loss => trigger={stop_price:.5f}")
    time.sleep(UI_DELAY)
    try:
        ouvrir_btn = wait.until(EC.element_to_be_clickable((By.XPATH, "//span[@data-testid='contract-trade-order-form-tab-open']")))
        time.sleep(UI_DELAY)
        ouvrir_btn.click()
        time.sleep(UI_DELAY)
    except Exception as e:
        print("[SL] Erreur clic 'Ouvrir':", e)
        try_click_confirmer_modal()
    try:
        stop_tab = wait.until(EC.element_to_be_clickable((By.XPATH, "//span[text()='Stop']")))
        time.sleep(UI_DELAY)
        stop_tab.click()
        time.sleep(UI_DELAY)
    except Exception as e:
        print("[SL] Erreur clic 'Stop':", e)
        try_click_confirmer_modal()
    try:
        trigger_input = wait.until(EC.element_to_be_clickable((By.XPATH, "//input[@placeholder='Prix de déclenchement']")))
        time.sleep(UI_DELAY)
        trigger_input.click()
        time.sleep(UI_DELAY)
        trigger_input.send_keys(Keys.CONTROL, "a")
        time.sleep(UI_DELAY)
        trigger_input.send_keys(Keys.BACKSPACE)
        time.sleep(UI_DELAY)
        trigger_input.send_keys(str(stop_price))
        time.sleep(UI_DELAY)
        print(f"[SL] Saisi du trigger price = {stop_price:.5f}")
    except Exception as e:
        print("[SL] Erreur saisie du stop price:", e)
    try:
        amount_input = driver.find_elements(By.XPATH, "//input[@autocomplete='off' and contains(@class,'ant-input') and @value='']")
        if amount_input:
            amt = amount_input[0]
            amt.click()
            time.sleep(UI_DELAY)
            amt.clear()
            time.sleep(UI_DELAY)
            amt.send_keys("100")
            time.sleep(UI_DELAY)
            print("[SL] Montant saisi = 100 (exemple)")
    except:
        pass
    scroll_bottom(driver)
    try_click_confirmer_modal()
    try:
        if is_long:
            btn_short = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "button[data-testid='contract-trade-open-short-btn']")))
            time.sleep(UI_DELAY)
            btn_short.click()
            time.sleep(UI_DELAY)
            print(f"[SL] Ordre Stop-Loss Long => SELL Trigger = {stop_price:.5f}")
        else:
            pass
    except Exception as e:
        print("[SL] Erreur clic 'Ouvrir Short' (Stop-Loss):", e)
        try_click_confirmer_modal()

def cancel_all_limit_orders():
    scroll_bottom(driver)
    try_click_confirmer_modal()
    print("[CANCEL] Annulation de tous les ordres...")
    time.sleep(UI_DELAY)
    try:
        open_orders_span = WebDriverWait(driver, 5).until(EC.element_to_be_clickable((By.XPATH, "//span[contains(text(),'Ordres ouverts')]")))
        time.sleep(UI_DELAY)
        open_orders_span.click()
        time.sleep(UI_DELAY)
    except Exception as e:
        print("[CANCEL] Erreur clic 'Ordres ouverts':", e)
        try_click_confirmer_modal()
    try:
        cancel_link = WebDriverWait(driver, 5).until(EC.element_to_be_clickable((By.XPATH, "//a[text()='Tout annuler']")))
        time.sleep(UI_DELAY)
        cancel_link.click()
        time.sleep(UI_DELAY)
        print("[CANCEL] 'Tout annuler' cliqué.")
    except Exception as e:
        print("[CANCEL] Erreur clic 'Tout annuler':", e)
        try_click_confirmer_modal()
    try:
        confirm_btn = WebDriverWait(driver, 5).until(EC.element_to_be_clickable((By.XPATH, "//button[@type='button' and contains(@class,'ant-btn-primary')]//span[text()='Confirmer']")))
        time.sleep(UI_DELAY)
        confirm_btn.click()
        time.sleep(UI_DELAY)
        print("[CANCEL] Annulation confirmée.")
    except Exception as e:
        print("[CANCEL] Erreur clic 'Confirmer' annulation:", e)
        try_click_confirmer_modal()

def close_position_immediately():
    scroll_bottom(driver)
    try_click_confirmer_modal()
    print("[CLOSE] Clôture instantanée.")
    time.sleep(UI_DELAY)
    try:
        close_btn = WebDriverWait(driver, 5).until(EC.element_to_be_clickable((By.XPATH, "//button[contains(.,'Clôture instantanée')]")))
        time.sleep(UI_DELAY)
        close_btn.click()
        time.sleep(UI_DELAY)
        print("[CLOSE] Position clôturée instantanément (market).")
    except Exception as e:
        print("[CLOSE] Erreur clic 'Clôture instantanée':", e)
        try_click_confirmer_modal()

def click_position_ouverte():
    scroll_bottom(driver)
    try_click_confirmer_modal()
    time.sleep(UI_DELAY)
    try:
        pos_open_span = WebDriverWait(driver, 5).until(EC.element_to_be_clickable((By.XPATH, "//span[contains(text(),'Position ouverte')]")))
        time.sleep(UI_DELAY)
        pos_open_span.click()
        time.sleep(UI_DELAY)
        print("[TP] 'Position ouverte(x)' cliqué.")
    except Exception as e:
        print("[TP] Erreur clic 'Position ouverte':", e)
        try_click_confirmer_modal()

def set_take_profit(tp_price):
    click_position_ouverte()
    try:
        add_modify_btn = WebDriverWait(driver, 5).until(EC.element_to_be_clickable((By.XPATH, "//span[text()='Ajouter' or text()='Modifier']")))
        time.sleep(UI_DELAY)
        add_modify_btn.click()
        time.sleep(UI_DELAY)
        print("[TP] Fenêtre Ajouter/Modifier ouverte.")
    except Exception as e:
        print("[TP] Erreur clic 'Ajouter/Modifier':", e)
        try_click_confirmer_modal()
    try:
        tp_input = WebDriverWait(driver, 5).until(EC.element_to_be_clickable((By.XPATH, "//div[contains(@class,'EditStopOrder_inputSectionTitle__') and contains(text(),'Take-Profit')]/ancestor::div[@class='ant-row']//span[contains(@class,'EditStopOrder_priceInput__')]//input[@class='ant-input']")))
        time.sleep(UI_DELAY)
        tp_input.click()
        time.sleep(UI_DELAY)
        tp_input.send_keys(Keys.CONTROL, "a")
        time.sleep(UI_DELAY)
        tp_input.send_keys(Keys.BACKSPACE)
        time.sleep(UI_DELAY)
        tp_input.send_keys(str(tp_price))
        time.sleep(UI_DELAY)
        print(f"[TP] TP saisi: {tp_price}")
    except Exception as e:
        print("[TP] Erreur saisie du TP:", e)
        try_click_confirmer_modal()
    try:
        confirm_btn = WebDriverWait(driver, 5).until(EC.element_to_be_clickable((By.XPATH, "//button[@type='button' and contains(@class,'ant-btn-primary')]//span[text()='Confirmer']")))
        time.sleep(UI_DELAY)
        confirm_btn.click()
        time.sleep(UI_DELAY)
        print("[TP] 1ère confirmation cliquée.")
    except Exception as e:
        print("[TP] Erreur 1er 'Confirmer' TP:", e)
        try_click_confirmer_modal()
    confirm_rappel_des_risques()

global_current_price = None

def poll_price():
    global global_current_price
    while True:
        price = get_doge_perp_price()
        if price is not None:
            global_current_price = price
        time.sleep(0.5)

price_thread = threading.Thread(target=poll_price, daemon=True)
price_thread.start()

def log_positions_csv():
    info = get_open_positions("SHIB_USDT")
    if not (info and "data" in info):
        return
    for pos in info["data"]:
        entry = pos.get("openAvgPrice")
        liq = pos.get("liquidatePrice")
        vol = pos.get("openVol")
        ts = int(time.time())
        print(f"CSV_POSITION;{ts};entry={entry};liq={liq};vol={vol}")

def log_orders_csv():
    info = get_open_orders("SHIB_USDT")
    if not (info and "data" in info):
        return
    ts = int(time.time())
    for order in info["data"]:
        order_id = order.get("orderId")
        price = order.get("price")
        vol = order.get("vol")
        o_type = order.get("orderType")
        status = order.get("stateText")
        print(f"CSV_ORDER;{ts};id={order_id};price={price};vol={vol};type={o_type};status={status}")

def main_strategy():
    print("=== DÉMARRAGE STRATÉGIE ===")
    market_size = 0.1
    limit_size  = 1.8
    tp_diff     = 0.000001
    while True:
        print("\n[STRAT] *** NOUVEAU CYCLE ***")
        log_positions_csv()
        log_orders_csv()
        pos_info = get_open_positions("SHIB_USDT")
        if (pos_info and "data" in pos_info and len(pos_info["data"]) > 0):
            print("[STRAT] Position existante => on la ferme pour un cycle propre.")
            close_position_immediately()
            time.sleep(UI_DELAY)
        cancel_all_limit_orders()
        time.sleep(UI_DELAY)
        open_market_order(str(market_size))
        time.sleep(EXCHANGE_DELAY)
        pos_info = get_open_positions("SHIB_USDT")
        log_positions_csv()
        if not (pos_info and "data" in pos_info and len(pos_info["data"]) > 0):
            print("[STRAT] Aucune position détectée après Market => on skip le cycle.")
            time.sleep(UI_DELAY)
            continue
        pos = pos_info["data"][0]
        entry_price = float(pos["openAvgPrice"])
        liquidation_price = float(pos["liquidatePrice"])
        print(f"[STRAT] Position => entry={entry_price:.10f}, liq={liquidation_price:.10f}")
        log_positions_csv()
        tp_value = entry_price + tp_diff
        set_take_profit(f"{tp_value:.10f}")
        time.sleep(UI_DELAY)
        limit1_price = liquidation_price * 1.005
        open_limit_order(str(limit_size), limit1_price)
        time.sleep(UI_DELAY)
        print(f"[STRAT] 1er Limit => {limit1_price:.10f}")
        log_orders_csv()
        limit1_filled = False
        while True:
            log_positions_csv()
            log_orders_csv()
            pos_info = get_open_positions("SHIB_USDT")
            if not (pos_info and "data" in pos_info and len(pos_info["data"]) > 0):
                print("[STRAT] Position fermée (TP ou manuel) => fin cycle.")
                break
            open_orders_info = get_open_orders("SHIB_USDT")
            limit1_still_open = False
            if (open_orders_info and "data" in open_orders_info):
                for o in open_orders_info["data"]:
                    if float(o.get("price", 0)) == float(limit1_price):
                        limit1_still_open = True
                        break
            if not limit1_still_open:
                print("[STRAT] 1er Limit n'est plus dans la liste => exécuté ou annulé.")
                limit1_filled = True
                break
            if global_current_price is not None and global_current_price >= tp_value:
                print(f"[STRAT] => TP atteint (prix={global_current_price:.10f} >= {tp_value:.10f}).")
                cancel_all_limit_orders()
                time.sleep(UI_DELAY)
                close_position_immediately()
                time.sleep(UI_DELAY)
                break
            time.sleep(UI_DELAY)
        pos_info = get_open_positions("SHIB_USDT")
        if not (pos_info and "data" in pos_info and len(pos_info["data"]) > 0):
            print("[STRAT] Position plus là => nouveau cycle.")
            continue
        if limit1_filled:
            cancel_all_limit_orders()
            time.sleep(UI_DELAY)
            pos_info = get_open_positions("SHIB_USDT")
            if not (pos_info and "data" in pos_info and len(pos_info["data"]) > 0):
                print("[STRAT] Position fermée => on repart.")
                continue
            pos = pos_info["data"][0]
            entry_price = float(pos["openAvgPrice"])
            liquidation_price = float(pos["liquidatePrice"])
            print(f"[STRAT] Nouvelle position => entry={entry_price:.10f}, liq={liquidation_price:.10f}")
            log_positions_csv()
            tp_value = entry_price + tp_diff
            set_take_profit(f"{tp_value:.10f}")
            time.sleep(UI_DELAY)
            limit2_price = liquidation_price * 1.005
            open_limit_order(str(limit_size), limit2_price)
            time.sleep(UI_DELAY)
            print(f"[STRAT] 2ème Limit => {limit2_price:.10f}")
            log_orders_csv()
            limit2_filled = False
            while True:
                log_positions_csv()
                log_orders_csv()
                pos_info = get_open_positions("SHIB_USDT")
                if not (pos_info and "data" in pos_info and len(pos_info["data"]) > 0):
                    print("[STRAT] Position fermée => fin cycle.")
                    break
                open_orders_info = get_open_orders("SHIB_USDT")
                limit2_still_open = False
                if (open_orders_info and "data" in open_orders_info):
                    for o in open_orders_info["data"]:
                        if float(o.get("price", 0)) == float(limit2_price):
                            limit2_still_open = True
                            break
                if not limit2_still_open:
                    print("[STRAT] 2ème Limit => plus dans la liste => exécuté/annulé.")
                    limit2_filled = True
                    break
                if global_current_price is not None and global_current_price >= tp_value:
                    print(f"[STRAT] => TP atteint (prix={global_current_price:.10f} >= {tp_value:.10f}).")
                    cancel_all_limit_orders()
                    time.sleep(UI_DELAY)
                    close_position_immediately()
                    time.sleep(UI_DELAY)
                    break
                time.sleep(UI_DELAY)
            pos_info = get_open_positions("SHIB_USDT")
            if not (pos_info and "data" in pos_info and len(pos_info["data"]) > 0):
                print("[STRAT] Position fermée => nouveau cycle.")
                continue
            if limit2_filled:
                cancel_all_limit_orders()
                time.sleep(UI_DELAY)
                pos_info = get_open_positions("SHIB_USDT")
                if not (pos_info and "data" in pos_info and len(pos_info["data"]) > 0):
                    print("[STRAT] Position fermée => on repart.")
                    continue
                pos = pos_info["data"][0]
                entry_price = float(pos["openAvgPrice"])
                liquidation_price = float(pos["liquidatePrice"])
                print(f"[STRAT] Position modifiée => entry={entry_price:.10f}, liq={liquidation_price:.10f}")
                log_positions_csv()
                tp_value = entry_price + tp_diff
                set_take_profit(f"{tp_value:.10f}")
                time.sleep(UI_DELAY)
                stop_price = entry_price * 0.98
                open_stop_loss(stop_price, is_long=True)
                time.sleep(UI_DELAY)
                log_orders_csv()
                while True:
                    log_positions_csv()
                    log_orders_csv()
                    pos_info = get_open_positions("SHIB_USDT")
                    if not (pos_info and "data" in pos_info and len(pos_info["data"]) > 0):
                        print("[STRAT] Position fermée => fin cycle complet.")
                        break
                    if global_current_price is not None and global_current_price >= tp_value:
                        print(f"[STRAT] => TP atteint (prix={global_current_price:.10f} >= {tp_value:.10f}).")
                        cancel_all_limit_orders()
                        time.sleep(UI_DELAY)
                        close_position_immediately()
                        time.sleep(UI_DELAY)
                        break
                    time.sleep(UI_DELAY)
                print("[STRAT] Fin du cycle => on relance un nouveau cycle.")
        time.sleep(UI_DELAY)
try:
    main_strategy()
except KeyboardInterrupt:
    print("[STRAT] Interrompu par l'utilisateur.")
finally:
    print("[STRAT] Fermeture du navigateur Selenium.")
    time.sleep(2)
    driver.quit()
