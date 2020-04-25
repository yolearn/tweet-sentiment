import time
def jaccard(str1, str2): 
    print(str1)
    print(str2)
    time.sleep(5)
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))