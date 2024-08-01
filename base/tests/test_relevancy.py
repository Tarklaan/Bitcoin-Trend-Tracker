from django.test import LiveServerTestCase
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import TimeoutException, WebDriverException
import time
from prettytable import PrettyTable

class ModelRelevancyTest(LiveServerTestCase):
    def setUp(self):
        try:
            service = Service('C:/Users/Mohsin/Desktop/FYP/100/chromedriver-win64/chromedriver-win64/chromedriver.exe')
            self.driver = webdriver.Chrome(service=service)
            self.driver.implicitly_wait(10)
        except WebDriverException as e:
            print(f"WebDriverException during setup: {e}")
            self.tearDown()

    def tearDown(self):
        if hasattr(self, 'driver'):
            self.driver.quit()

    def test_prediction_page(self):
        self.driver.get(self.live_server_url + '/predict/')
        try:
            # Wait for the table to be present
            WebDriverWait(self.driver, 20).until(
                EC.presence_of_element_located((By.ID, 'prediction-table'))
            )
            table = self.driver.find_element(By.ID, 'prediction-table')
            table_rows = table.find_elements(By.TAG_NAME, 'tr')

            # Use PrettyTable to format the table data
            pretty_table = PrettyTable()

            for i, row in enumerate(table_rows):
                cells = row.find_elements(By.TAG_NAME, 'td') or row.find_elements(By.TAG_NAME, 'th')
                row_data = [cell.text for cell in cells]
                if i == 0:  # Header row
                    pretty_table.field_names = row_data
                else:
                    pretty_table.add_row(row_data)

            print("Predictions for Bitcoin (USD$):")
            print(pretty_table)

            # Ensure all rows in the table have data
            for row_data in pretty_table._rows:
                self.assertTrue(all(row_data))

            print("=============================++++++++++++++++++==============================")
            investment_advice = self.driver.find_element(By.ID, 'investment-advice').text
            print("Investment Advice:", investment_advice)
        except (TimeoutException, WebDriverException) as e:
            print(f"An error occurred in test_prediction_page: {e}")
            print(self.driver.page_source)  # Debugging: print the page source

    def test_other_crypto_page(self):
        self.driver.get(self.live_server_url + '/otherCrypto/')
        try:
            # Wait for the crypto items to be present
            WebDriverWait(self.driver, 20).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, '.crypto-item'))
            )
            crypto_items = self.driver.find_elements(By.CSS_SELECTOR, '.crypto-item')
            if crypto_items:
                for i in range(min(6, len(crypto_items))):
                    crypto_items[i].click()
                    # Add a small delay to allow the click action to register
                    time.sleep(0.5)

            # Wait for the selected prices table to be present
            WebDriverWait(self.driver, 20).until(
                EC.presence_of_element_located((By.ID, 'selected-prices-table'))
            )
            table = self.driver.find_element(By.ID, 'selected-prices-table')
            table_rows = table.find_elements(By.TAG_NAME, 'tr')

            # Use PrettyTable to format the table data
            pretty_table = PrettyTable()

            for i, row in enumerate(table_rows):
                cells = row.find_elements(By.TAG_NAME, 'td') or row.find_elements(By.TAG_NAME, 'th')
                row_data = [cell.text for cell in cells]
                if i == 0:  # Header row
                    pretty_table.field_names = row_data
                else:
                    pretty_table.add_row(row_data)

            print("Selected Prices Table Data (USD$):")
            print(pretty_table)

            # Ensure all rows in the table have data
            for row_data in pretty_table._rows:
                self.assertTrue(all(row_data))

            print("=============================++++++++++++++++++==============================")
            investment_advice = self.driver.find_element(By.ID, 'investment-advice').text
            print("Investment Advice:", investment_advice)
        except (TimeoutException, WebDriverException) as e:
            print(f"An error occurred in test_other_crypto_page: {e}")
            print(self.driver.page_source)  # Debugging: print the page source

if __name__ == "__main__":
    LiveServerTestCase().run()
