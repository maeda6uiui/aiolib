import hashlib

def get_md5_hash(v:str):
    return hashlib.md5(v.encode()).hexdigest()
