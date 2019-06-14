def get_header(filename, cmt=False):
    """Return header (and comments) from flat file"""
    flat = open(filename).readlines()

    header = {}
    comment = {}

    for line in flat:
        if len(line.strip()) == 0:
            continue
        else:
            key, end = [part.strip() for part in line.split("=", 1)]
            seq = end.split(" /", 1)
            if len(seq) == 1:
                seq = end.split("\t/", 1)
            val = seq[0].strip().strip("'").strip()
            if len(seq) == 2:
                com = seq[1].strip()
            else:
                com = "no comment"
        
        if val.isdecimal():
            val = int(val)
        elif val.replace(".", "").lstrip("-").lstrip("+").isnumeric():
            val = float(val)
            
        header[key] = val
        comment[key] = com
    
    if cmt:
        return header, comment
    else:
        return header
