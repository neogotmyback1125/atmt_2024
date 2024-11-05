from subword_nmt import learn_bpe

# 指定输入文件和输出文件
input_file = 'data/en-fr/preprocessed/train.fr'
output_file = 'data/en-fr/preprocessed/bpe.codes'

# 设置 BPE 操作次数
num_operations = 10000

# 打开输入文件和输出文件，使用 errors='ignore' 来忽略无法解码的字符
with open(input_file, 'r', encoding='utf-8', errors='ignore') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
    learn_bpe.learn_bpe(infile, outfile, num_operations)
