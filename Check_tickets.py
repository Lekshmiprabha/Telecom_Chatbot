import sqlite3
conn = sqlite3.connect("data/tickets.db")
cur = conn.cursor()
cur.execute("SELECT COUNT(*) FROM tickets")
print(cur.fetchone())
conn.close()