from numtotext import NumToVnStr
import re
import numpy as np

custom_converter = NumToVnStr(đọc_số_rỗng=False, linh='lẻ', tư='bốn', nghìn='nghìn', mươi="mươi", tỷ='tỉ', lăm='lăm')
with open("convert.txt", 'r', encoding='utf-8') as f:
    convert = f.read().strip().split("\n")


with open("units.txt", 'r', encoding='utf-8') as f:
    units = f.read().strip().split('\n')

dict_convert = {}
for line  in convert:
    out = re.split(':', line)
    out[0] = ' ' + out[0] + ' '
    out[1] = ' ' + out[1] + ' '
    dict_convert[out[0]] = out[1]

dict_unit = {}
for line in units:
    out = re.split(':', line)
    out[0] = out[0].strip()
    out[1] = out[1].strip()
    dict_unit[out[0]] = out[1]
    
def replace_keyword(text):
    for key in dict_convert:
        text = re.sub(key, dict_convert[key], text)
    return text

def findURL(text):
    pattern = "(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})"
    
    urls = re.findall(pattern, text)
    
    return urls

def patternUrl2Text(text):
    urls = findURL(text)
    
    print("Url", urls)
    
    for u in urls:
        text = re.sub(u, "linh đính kèm", text)
    
    return text

# pattern 1,5
def findPatternComma(text):
    pattern = r'\d,\d'
    
    comma = re.findall(pattern, text)
    
    return comma

def patternComma2Text(text):
    comma = findPatternComma(text)
    
    for c in comma:
        c_process = re.sub(",", " phẩy ", c)
        text = re.sub(c, c_process, text)

    return text

# pattern J717
def findPatternJ(text):
    pattern = "m\d+|j\d+|a\d+|s\d+"
    results = re.findall(pattern, text)
    
    return results

def patternJ2Text(text):
    patternJ = findPatternJ(text)
    
    for p in patternJ:
        num = re.findall('\d+', p)
        char = re.findall('m|j|a|s', p)
        
        p_process = char[0] + ' ' + num[0]
        text = re.sub(p, p_process, text)
    
    return text

# Pattern unit
def findPatternUnit(text):
    pattern = "\d+\s*nm[\.\,\s]+|\d+\s*gb[\.\,\s]+|\d+\s*hz[\.\,\s]+|\d+\s*%[\.\,\s]+|\d+\s*g[\.\,\s]+|\d+\s*k[\.\,\s]+|\d+\s*km[\.\,\s]+|\d+\s*kg[\.\,\s]+|\d+\.*\d+\s*nits[\.\,\s]+|\d+\.*\d+\s*mah[\.\,\s]+|\d+\s*tb/s[\.\,\s]+|\d+\,*\d+\s*mb/s[\.\,\s]+|\d+\s*tb[\.\,\s]+|\d+\s*fps[\.\,\s]+|\d+\s*mm[\.\,\s]+|\d+\s*mb[\.\,\s]+|\d+\s*x[\.\,\s]+"
    results = re.findall(pattern, text)
    
    return results

def patternUnit2Text(text):
    patternUnit = findPatternUnit(text)
    
    for p in patternUnit:
        num = re.findall('\d+\.*\d*', p)
        unit = re.findall('nm|gb|hz|%|g|k|km|kg|nits|mah|tb/s|mb/s|tb|fps|mm|mb|x', p)
    
        p_process = num[0] + ' ' + dict_unit[unit[0]]
        text = re.sub(p, p_process, text)
    
    return text

# pattern %
def findPatternPercent(text):
    pattern = r'\d\s*%'
    
    percent = re.findall(pattern, text)
    
    return percent

def patternPercent2Text(text):
    percent = findPatternPercent(text)
    
    for p in percent:
        p_process = re.sub("%", " phần trăm ", p)
        text = re.sub(p, p_process, text)
    
    return text

# pattern date dd/mm
def findpatternDateDDMM(text):
    pattern = r'\d{1,2}/\d{1,2}'

    ddmm = re.findall(pattern, text)
    
    return ddmm

def patternDateDDMM2Text(text):
    ddmm = findpatternDateDDMM(text)
    
    for dm in ddmm:
        dm_process = " ngày " + dm
        dm_process = re.sub('/', " tháng ", dm_process)
        text = re.sub(dm, dm_process, text)
    
    return text

# Pattern dd/mm/yyyy
def findPatternDDMMYYYY(text):
    pattern = r'\d{1,2}/\d{1,2}/\d{4}'
    
    ddmmyyyy = re.findall(pattern, text)
    
    return ddmmyyyy

def patternDDMMYYYY2Text(text):
    ddmmyyyy = findPatternDDMMYYYY(text)
    
    for d in ddmmyyyy:
        d_split = d.split('/')
        d_process = " ngày " + str(d_split[0]) + " tháng " + str(d_split[1]) + " năm " + str(d_split[2])
        text = re.sub(d, d_process, text)
    
    return text

# Pattern mm/yyyy
def findPatterMMYYYY(text):
    pattern = r'\d{1,2}/\d{4}'
    
    mmyyyy = re.findall(pattern, text)
    
    return mmyyyy

def patternMMYYYY2Text(text):
    mmyyyy = findPatterMMYYYY(text)
    
    for d in mmyyyy:
        d_split = d.split('/')
        d_process = str(d_split[0]) + " năm " + str(d_split[1])
        text = re.sub(d, d_process, text)        
    return text

# Pattern 5h, 6h
def findPatternHour(text):
    pattern = r'\d{1,2}[\.\,\d{1,2}]*h'

    hours = re.findall(pattern, text)

    return hours

def patternHours2Text(text):
    hours = findPatternHour(text)

    for h in hours:
        number = re.findall('\d+', h)
        if len(number) == 1:
            h_process = str(int(number[0])) + " giờ "
            text = re.sub(h, h_process, text)
        elif len(number) == 2:
            h_process = str(int(number[0])) + " giờ " + str(int(number[1]))
            text = re.sub(h, h_process)
    
    return text


# Pattern d-d năm
def findPatternDtoD(text):
    pattern = r'\d\s*-\s*\d\s*'
    
    dd = re.findall(pattern, text)
    
    return dd

def patternDtoD2text(text):
    dd = findPatternDtoD(text)
    
    for d in dd:
        d_process = re.sub('\s*-\s*', " đến ", d)
        text = re.sub(d, d_process, text)
    
    return text

# Pattern find number dot 100.000
def findNumberDot(text):
    pattern = r'\d+[.]\d+'
    
    dot = re.findall(pattern, text)
    
    return dot

def numberDot2Text(text):
    dot = findNumberDot(text)
    
    for d in dot:
        d_process = re.sub('[.]', '', d)
        text = re.sub(d, d_process, text)
    
    return text

# Pattern find number 
def findNumbers(text):
    pattern = r'\d+'

    numbers = re.findall(pattern, text)

    len_numbers = [len(a) for a in numbers]
    idx_num = np.argsort(len_numbers)

    output = [numbers[i] for i in idx_num]
    output.reverse()

    return output
def number2Text(text):
    nums = findNumbers(text)
    
    for n in nums:
        n_process = custom_converter.to_vn_str(n)
        text = re.sub(n, ' ' + n_process, text)
    
    return text

# Remove character specific
def removeCharacterSpecific(text):
    text = re.sub('[\/\-\*\(\)\[\]\"\'\=\+\&\%\$\#\@\?\<\>\…]', " ", text)
    # text = re.sub('\s+', ' ', text)
    return text
    

def preprocess_text(text):
    text = replace_keyword(text)
    text = patternUrl2Text(text)
    text = patternComma2Text(text)
    text = patternJ2Text(text)
    text = patternUnit2Text(text)
    text = patternPercent2Text(text)
    text = patternDDMMYYYY2Text(text)
    text = patternDateDDMM2Text(text)
    text = patternMMYYYY2Text(text)
    text = patternHours2Text(text)
    text = numberDot2Text(text)
    text = number2Text(text)

    # text = removeCharacterSpecific(text)
    # print("Text character: ", text)
    
    return text
        
if __name__=='__main__':
    text = "Cụ thể, Pháp có 12 tiểu đoàn, 7 đại đội bộ binh (trong quá trình chiến dịch được tăng viện 4 tiểu đoàn và 2 đại đội lính nhảy dù), 2 tiểu đoàn pháo binh 105mm (24 khẩu - sau đợt 1 được tăng thêm 4 khẩu nguyên vẹn và cho đến ngày cuối cùng được thả xuống rất nhiều bộ phận thay thế khác), 1 đại đội pháo 155mm (4 khẩu), 2 đại đội súng cối 120mm (20 khẩu), 1 tiểu đoàn công binh, 1 đại đội xe tăng 18 tấn (10 chiếc M24 Chaffee của Mỹ), 1 đại đội xe vận tải 200 chiếc, 1 phi đội máy bay gồm 14 chiếc (7 máy bay khu trục, 6 máy bay liên lạc trinh sát, 1 máy bay lên thẳng)."
    # print(patternPercent2Text(patternComma2Text(text)))
    text = preprocess_text(text)
    print(text)