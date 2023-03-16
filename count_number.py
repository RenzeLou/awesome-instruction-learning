import re
import requests
from bs4 import BeautifulSoup

with open("README.md", "r", encoding='utf-8') as f_in:
    buf = f_in.readlines()
    count = 0
    sec_count = 0
    count_flag = False
    table_flag = False
    sec_title = None
    for line in buf:
        if "Table of Contents" in line:
            count_flag = True  # start counting
            continue
        if count_flag:
            if line.startswith("#"):
                # indicate a new section
                # then print the previous section's number
                if sec_title is not None:
                    print("==>", sec_title, "\nsec paper number: ", sec_count, "\n")
                    sec_count = 0
                sec_title = line.strip()
                # note the "Corpora" section
                table_flag = False
                if "Corpora" in line:
                    table_flag = True
            if re.match(r'^\d+\.\s+\*\*', line) and not table_flag:
                count += 1
                sec_count += 1
            elif table_flag: 
                # since the "Corpora" section is a table, so we need to count it seperately
                repo_url = 'https://github.com/RenzeLou/awesome-instruction-learning'
                readme_path = 'README.md'

                # make a GET request to the raw content URL of the README.md file
                raw_url = f'{repo_url}/raw/main/{readme_path}'
                response = requests.get(raw_url)
                # parse the HTML content using BeautifulSoup
                soup = BeautifulSoup(response.text, 'html.parser')
                # find the table element by its class or ID
                table = soup.find('table', {'class': '"copora-table"'})  # replace with the actual class or ID of your table
                # count the number of rows in the table
                if table:
                    rows = table.find_all('tr')
                    num_rows = len(rows)
                else:
                    # throw a warning if the table was not found
                    Warning(f'Table not found in {readme_path}')
                    num_rows = 0
                
                count += num_rows - 1
                sec_count += num_rows - 1
            
print("\nTotal paper number: ", count)