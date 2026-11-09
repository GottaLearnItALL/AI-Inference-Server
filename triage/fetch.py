import requests, os, json
from dotenv import load_dotenv
import sqlite3
from datetime import datetime, timezone
from triage.db import DB_PATH, init_db

load_dotenv()
init_db(DB_PATH)
connection = sqlite3.connect(DB_PATH)
cursor = connection.cursor()

endpoint = 'https://api.github.com/repos/langchain-ai/langchain/issues'

def fetch_all():

    page = 1
    skipped = 0
    while True:
        if page > 2: 
            break
        try:
            data = requests.get(endpoint,
            params={
                "state": "open",
                "per_page":100,
                #"page":page
                },
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {os.environ.get('GIT_TOKEN')}"
            })
            data.raise_for_status()

            batch = data.json()
            if not batch:
                break
                
            for issues in batch:
                if "pull_request" in issues:
                    skipped+=1
                    continue
                
                title = issues['title']
                issue_id = issues["number"]
                state = issues['state']
                created_at = issues['created_at']
                labels = json.dumps([l["name"] for l in issues["labels"]])
                body = issues['body'] or ""
                print(f"Fetched {title}")
                upsert_query = "INSERT OR REPLACE INTO issues (number, title, body, labels, state, created_at, fetched_at) VALUES (?,?,?,?,?,?,?)"
                try:
                    cursor.execute(upsert_query, (issue_id, title, body, labels, state, created_at ,datetime.now(timezone.utc).isoformat()))
                except Exception as e:
                    print(f"Exception {e} occured")
                    break
            
            
            page+=1
            connection.commit()

        except Exception as e:
            print(f"Erorr {e} occured while fetching!")
            break


    print(skipped)
    connection.close()

if __name__ == "__main__":
    fetch_all()


