import csv

# 读取 data 文件
with open('letter-recognition.data', 'r') as infile:
    lines = infile.readlines()

# 写入 CSV 文件
with open('letter-recognition.csv', 'w', newline='') as outfile:
    writer = csv.writer(outfile)
    
    # 写入表头
    header = ['letter'] + [f'feature_{i+1}' for i in range(16)]
    writer.writerow(header)
    
    # 写入数据行
    for line in lines:
        line = line.strip()
        if line:
            row = line.split(',')
            writer.writerow(row)

print(f'转换完成！共写入 {len(lines)} 行数据到 letter-recognition.csv')
