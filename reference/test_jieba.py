import jieba

text = "本公司有部分普通发票商品销售发票增值税发票及其他服务行业发票"
print("源文本: ", text)

# 精确模式
seg_list = jieba.cut(text)
print("精确模式: ", " ".join(seg_list))

# 全模式
seg_list = jieba.cut(text, cut_all=True)
print("全模式: ", " ".join(seg_list))

# 搜索引擎模式
seg_list = jieba.cut_for_search(text)
print("搜索引擎模式: ", " ".join(seg_list))
