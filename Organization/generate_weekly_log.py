from datetime import datetime, timedelta
import os

# Calcola il lunedì della settimana corrente
today = datetime.today()
start_of_week = today - timedelta(days=today.weekday())  # lunedì
week_number = start_of_week.isocalendar().week
date_str = start_of_week.strftime("%d-%m_%Y")

# Nome del file
filename = f"week_{week_number}_{date_str}.md"

# Path della cartella "Logs" (nello stesso livello del file .py)
current_dir = os.path.dirname(os.path.abspath(__file__))
logs_dir = os.path.join(current_dir, "..", "Logs")
os.makedirs(logs_dir, exist_ok=True)  # crea Logs se non esiste

file_path = os.path.join(logs_dir, filename)

# Template
template = f"""# 📘 Thesis Weekly Progress Report

## Week: {start_of_week.year} - Week {week_number}
## Dates: From {start_of_week.strftime('%d-%m-%Y')} to {(start_of_week + timedelta(days=6)).strftime('%d-%m-%Y')}

---

### 🎯 Weekly Goals
- [ ] Goal 1
- [ ] Goal 2
- [ ] Goal 3

---

### 📅 Daily Log

#### 📆 Monday
- **Main Focus:** 
- **What I did:** 
- **Challenges:** 
- **Time spent:** 

#### 📆 Tuesday
- **Main Focus:** 
- **What I did:** 
- **Challenges:** 
- **Time spent:** 

#### 📆 Wednesday
- **Main Focus:** 
- **What I did:** 
- **Challenges:** 
- **Time spent:** 

#### 📆 Thursday
- **Main Focus:** 
- **What I did:** 
- **Challenges:** 
- **Time spent:** 

#### 📆 Friday
- **Main Focus:** 
- **What I did:** 
- **Challenges:** 
- **Time spent:** 

#### 📆 Saturday / Sunday (optional)
- **Catch-up / Reading / Breaks:**

---

### ✅ Summary of Work Done
- Key sections completed
- Key insights or improvements
- Progress vs goals

---

### ⚠️ Issues / Questions to Resolve
- Any open questions or blockers
- Things to ask the advisor

---

### 🔜 Goals for Next Week
- [ ] Goal 1
- [ ] Goal 2
"""

# Scrive il file solo se non esiste già
if not os.path.exists(file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(template)
    print(f"✅ Created: {file_path}")
else:
    print(f"⚠️ File already exists: {file_path}")
