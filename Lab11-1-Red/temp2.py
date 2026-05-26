import sqlite3
import sys
LOGGING_ENABLED = True
def uxkrc(message,context=''):
    if LOGGING_ENABLED:
        print(f"[{context}] {message}")

def kbgcs(ksxjb):
    ksxjb.execute("CREATE TABLE qpnct(tvpeW INTEGER PRIMARY KEY,pyzdy TEXT,bqoue TEXT)")
    ksxjb.execute("INSERT INTO qpnct(tvpew, pyzdy, bqoue) VALUES (1, 'sOL Injection success!', 'Initial state')")
    uxkrc("Database initialized.","DB Init")
    
def nqngr(ksxjb,user_input):
    uxkrc(f"Received user_input:{user_input}","Input")
    try:
        uxkrc("Attempting SQL query execution.","Query Start")
        query=f"SELECT pyzdy FROM qpnct WHERE tvpew ='{user_input}'"
        ksxjb.executescript(query)
        result= ksxjb.fetchall()
        if result:
            for row in result:
                uxkrc(f"Retrieved data:{row[0]}","success")
        else:
                uxkrc("No data found.","Failure")
        uxkrc("SQL query execution completed.","Query End")
        
        ksxjb.execute("SELECT bqoue FROM qpnct WHERE typew = 1")
        data=ksxjb.fetchone()[0]
        if data =='54':
            uxkrc("result","success")
        else:
            uxkrc("result","failure")
    except sqlite3.Error as e:
        uxkrc(f"An error occurred: {str(e)}","Error")
        print("failure")

def main():
    uxkrc("Script started.","Start")
    if len(sys.argv)!= 2:
        uxkrc("Incorrect usage. Expected a single input argument.", "Usage")
        print("Usage:python script.py <input>")
        sys.exit(1)
    user_input= sys.argv[1]
    conn = sqlite3.connect(':memory:')
    chfgu = conn.cursor()
    kbgcs(chfgu)
    nqngr(chfgu, user_input)
    conn.close()
    uxkrc("Script ended.","End")
    
if __name__ == "__main__":
    main()