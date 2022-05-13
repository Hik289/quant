import numpy as np
import pandas as pd
import openpyxl
from openpyxl.chart import Reference, LineChart, BarChart, Series


#-----------------！！！画图！！！----------------------------------
#------------------------------------------------------------------------
wb = openpyxl.load_workbook('Increase&Decrese_%s.xlsx' %y_str)
mrow = new_icrs.shape[0] + 1  #max_row

for ws_title in ['增持-all', '减持-all']:
    ws = wb[ws_title]
    start_rows = [4,7,11]
    chart_titles = ['%s股票个数占比'%(ws_title[0:2]), '%s金额'%(ws_title[0:2]), '%s金额/总市值'%(ws_title[0:2])]
    
    for i in range(3): #三张图
        #条状图
        chart10 = BarChart()
        data10 = Reference(ws, min_col = start_rows[i], min_row = 1, max_row = mrow)
        chart10.add_data(data10, titles_from_data=True)
        dates = Reference(ws, min_col=1, min_row=2, max_row=mrow)
        chart10.set_categories(dates)
        chart10.title = chart_titles[i]

        chart11 = LineChart()
        data11 = Reference(ws, min_col=start_rows[i]+1, min_row=1, max_col=start_rows[i]+2, max_row=mrow)
        chart11.add_data(data11, titles_from_data=True)
        chart11.y_axis.axId = 200
        chart11.y_axis.crosses = 'max'  
        
        chart10 += chart11
        chart10.width = 30
        chart10.height = 10
        ws.add_chart(chart10, "o"+str(20*i+2))

wb.save('Increase&Decrese_%s.xlsx' %y_str)
