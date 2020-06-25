nations = """ALGERIA
ARGENTINA
BRAZIL
CANADA
EGYPT
ETHIOPIA
FRANCE
GERMANY
INDIA
INDONESIA
IRAN
IRAQ
JAPAN
JORDAN
KENYA
MOROCCO
MOZAMBIQUE
PERU
CHINA
ROMANIA
SAUDI ARABIA
VIETNAM
RUSSIA
UNITED KINGDOM
UNITED STATES
"""
nations = nations.split('\n')

regions = """AFRICA
AMERICA
ASIA
EUROPE
MIDDLE EAST
"""
regions = regions.split('\n')

# process suppliers
lines = open('supplier.tbl').readlines()
o = []
for line in lines:
  try:
    parts = line.split('|')
    parts[4] = str(nations.index(parts[4]))
    parts[5] = str(regions.index(parts[5]))
    parts[3] = str(int(parts[4]) * 10 + int(parts[3][-1]))
    o.append('|'.join(parts))
  except:
    print line
    break

f = open('supplier.tbl.p','w')
for line in o:
  f.write(line)
f.close()

# process customers
lines = open('customer.tbl').readlines()
o = []
for line in lines:
  try:
    parts = line.split('|')
    parts[4] = str(nations.index(parts[4]))
    parts[5] = str(regions.index(parts[5]))
    parts[3] = str(int(parts[4]) * 10 + int(parts[3][-1]))
    o.append('|'.join(parts))
  except:
    print line
    break

f = open('customer.tbl.p','w')
for line in o:
  f.write(line)
f.close()

# process parts
lines = open('part.tbl').readlines()
o = []
for line in lines:
  try:
    parts = line.split('|')
    parts[2] = parts[2].split('#')[-1]
    parts[3] = parts[3].split('#')[-1]
    parts[4] = parts[4].split('#')[-1]
    o.append('|'.join(parts))
  except:
    print line
    break

f = open('part.tbl.p','w')
for line in o:
  f.write(line)
f.close()

