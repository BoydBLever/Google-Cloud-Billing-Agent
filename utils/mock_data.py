import json
import os

MOCK_DIR = os.path.join(os.path.dirname(__file__), "..", "mock")

def load_json(name: str):
    path = os.path.join(MOCK_DIR, name)
    with open(path, "r") as f:
        return json.load(f)

def save_json(name: str, data):
    path = os.path.join(MOCK_DIR, name)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def get_account(account_id: str):
    accounts = load_json("accounts.json")
    return accounts.get(account_id)

def get_policy(policy_key: str):
    policies = load_json("policies.json")
    return policies.get(policy_key)

def create_ticket(account_id: str, issue: str):
    tickets = load_json("tickets.json")
    tickets["tickets"].append({
        "account": account_id,
        "issue": issue,
        "generated_on": __import__("datetime").datetime.utcnow().isoformat() + "Z"
    })
    save_json("tickets.json", tickets)
    return "Ticket created."
