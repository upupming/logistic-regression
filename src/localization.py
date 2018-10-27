from matplotlib import pyplot

# 用来正常显示中文标签
pyplot.rcParams['font.family'] = 'sans-serif'
pyplot.rcParams['font.sans-serif'] = ['SimHei', 'Helvetica', 'Calibri']
# 用来正常显示负号
pyplot.rcParams['axes.unicode_minus'] = False