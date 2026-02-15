import pdb
import copy
import folium
import random
import pickle
import  numpy as np
import matplotlib.pyplot as plt
import numpy as np
import random
import time
from sklearn.cluster import KMeans
import numpy as np
import  bisect

from shapely.geometry import LineString



def transfer_status_to_num(string1):
    if string1=='畅通':
        return 0
    elif string1=='缓行':
        return 1
    elif string1=='拥堵':
        return 2
#输入站点给出经纬度信息函数
# args:只根据名字对应线路的字典，方向，名字，索引
def Stationpair_get_latlon(name_to_Stationlatlonlist,dire,name,index):
    index=int(index)
    # print(name_to_Stationlatlonlist.keys())
    lines=name_to_Stationlatlonlist['"{}"'.format(name)]
    if dire=='0':
        return [lines[0][index-1],lines[1][index-1]]
    if dire=='1':
        return [lines[2][index-1],lines[3][index-1]]



def is_short_line_almost_on_long_line(short_line_coords, long_line_coords, tolerance=1e-6):
    # 创建 LineString 对象
    short_line = LineString(short_line_coords)
    long_line = LineString(long_line_coords)

    # 为长线创建缓冲区
    long_line_buffer = long_line.buffer(tolerance)


    # 检查短线是否完全在长线的缓冲区内
    if long_line_buffer.contains(short_line):

        return True

    # 检查交集长度是否足够长
    intersection = short_line.intersection(long_line_buffer)
    if intersection.length >= short_line.length * (1 - tolerance):
        return True

    return False
def find_max_value(lst):
    """
    查找列表中的最大值
    
    参数:
        lst (list): 输入的数字列表
        
    返回:
        最大值，如果列表为空则返回None
    """
    if not lst:
        return None
    
    max_val = lst[0]
    for val in lst:
        if val > max_val:
            max_val = val
    
    return max_val
#随机产生颜色代码
def generate_random_color():
    return f"#{random.randint(0, 0xFFFFFF):06x}"

def remove_zero(li):
    filtered_numbers = [num for num in li if num != 0]
    return  filtered_numbers


def random_expend_list(original_list, triplist,N,tree_,tree_to_vehicle,Faster=False):
    
    current_time=time.time()
    new_list = original_list.copy()
    return_list=[]
    if len(triplist)==0:
        return new_list,[],tree_
   
        
    if N>len(triplist):
        N=len(triplist)
        
    if Faster:
        N=int(2*N)
    for _ in range(N):
        random_element = triplist[random.randint(0,len(triplist)-1)]
        
        if random_element in tree_to_vehicle.keys():
            new_vehicle = copy.deepcopy(tree_to_vehicle[random_element])
            idiop=new_vehicle.get('id')
       
            new_id = get_unique_id(new_list,idiop)
            new_list.append(new_id)
            return_list.append(new_id)

            new_vehicle.set('id', new_id)
            # new_vehicle.set('depart', '0')

            # 将新的vehicle添加到self.sumo_roudata中
            tree_.append(new_vehicle)
   
    # print(original_list)
    # print(new_list)
    return new_list,return_list,tree_

def random_reduce_list(original_list, triplist,N,tree_,tree_to_vehicle,Faster=False):
    current_time=time.time()
    new_list = original_list.copy()
    return_list=[]
    if len(triplist)==0:
        return new_list,[],tree_
    if abs(N)>len(triplist):
        N=len(triplist)
    if Faster:
        triplist=original_list[:int(len(original_list)/2)+1]
        N=len(triplist)

    for _ in range(abs(N)):
        if len(new_list) <= 0:
            return new_list,return_list,tree_
        
        return_list.append(triplist[_])
        
        
        if triplist[_] in tree_to_vehicle.keys() :
     
            
            tree_.remove(tree_to_vehicle[triplist[_]])
            new_list.remove(triplist[_])
   
    return new_list,return_list,tree_
def get_unique_id(id_list, new_id):
    """
    生成一个唯一的ID，如果ID已存在则在前面加's'
    
    参数:
    id_list: 现有的ID列表
    new_id: 需要检查的新ID
    
    返回:
    str: 唯一的ID
    """
    while new_id in id_list:
        new_id = 's' + new_id
    return new_id

#画图_line函数
def draw_line(city,drawline_list,filename):
    #Jia_Xing,Tianjin
    with open('F:\TianJinData\新建文件夹\dict_name_latlonlist_{}.pkl'.format(city), 'rb') as file:
        dict_name_to_latlonlist = pickle.load(file)
    lines = dict_name_to_latlonlist
    m = folium.Map(location=[39.9087, 116.3975], zoom_start=13)
    colors = [generate_random_color() for _ in range(len(lines))]

    # 遍历字典，绘制每条线路

    line_draw={}
    for k in drawline_list:
        line_draw[k]=lines[k]

    for idx, (line_name, coordinates) in enumerate(line_draw.items()):
        latitudes = coordinates[0]
        longitudes = coordinates[1]
        combined_coordinates = list(zip(latitudes, longitudes))

        folium.PolyLine(
            locations=combined_coordinates,  # 经纬度坐标列表
            color=colors[idx],  # 使用颜色列表中的颜色
            weight=5,  # 线条宽度
            opacity=0.8,  # 线条透明度
            tooltip=line_name  # 鼠标悬停时显示的线路名称
        ).add_to(m)

    m.save('Bad_case_distribution_lines_{}.html'.format(filename))





def draw_point(city,drawpoint_list,filename):
    #Jia_Xing,Tianjin
    with open('F:\TianJinData\新建文件夹\dict_name_latlonlist_{}.pkl'.format(city), 'rb') as file:
        dict_name_to_latlonlist = pickle.load(file)
    lines = dict_name_to_latlonlist
    m = folium.Map(location=[39.9087, 116.3975], zoom_start=13)
    colors = [generate_random_color() for _ in range(len(lines))]

    # 遍历字典，绘制每条线路


    list_latlon=[]
    for k in drawpoint_list:
        testid = k
        direc, name, index = testid.split('_')
        listitem = Stationpair_get_latlon(dict_name_to_latlonlist,direc, name, index)
        list_latlon.append(listitem)

    for point in list_latlon:
        folium.Marker(
            location=point,  # 经纬度坐标
            popup=f"Point: {point}",  # 点击时显示的弹出信息
            icon=folium.Icon(color='blue')  # 设置点的颜色和图标
        ).add_to(m)

    m.save('Bad_case_distribution_points_{}.html'.format(filename))

#计算时序的阈值列表
def calculate_changes(time_series):
    change_list=[]
    for i in range(4,len(time_series)):
        if time_series[i - 1] == 0:
            continue

        # 计算当前时刻加上之前两个时刻的平均值
        curr_avg=np.mean([ time_series[k] for k in range(i-2,i+1) if time_series[k]!=0])
        pre_avg=np.mean([ time_series[k] for k in range(i-3,i) if time_series[k]!=0])
        if curr_avg<=pre_avg:
            continue
        else:
            change_list.append((curr_avg-pre_avg)/pre_avg)


    return change_list



# 计算单个站点不同时刻的rmse
def calculate_rmse_per_station(pred, true):
    rmse_per_station = {}
    pre_mask,true_mask=pred,true
    # meanrmse= np.sqrt(np.mean((pre_mask - true_mask) ** 2))

    # 遍历每个站点
    for station in range(pred.shape[0]):

        # 获取当前站点的预测值、真实值和掩码
        pred_station = pred[station]
        true_station = true[station]
        # mask_station = mask[station]

        # 筛选出有效元素
        # valid_indices = np.where(mask_station)
        pred_valid = pred_station
        true_valid = true_station

        # 计算差值平方
        diff_squared = (pred_valid - true_valid) ** 2
        diff_squared=np.nan_to_num(diff_squared)

        # 计算RMSE
        rmse = np.sqrt(diff_squared)

        # 存储当前站点的RMSE
        rmse_per_station[station] = rmse

    return rmse_per_station


# 计算一个序列的acf是否够高
def calculate_acf(sequence, max_lag):
    """
    手动计算时间序列的自相关函数（ACF），处理序列中的 NaN 值。
    """
    N = len(sequence)

    # 计算均值
    mean = np.nanmean(sequence)

    # 计算方差
    variance = np.nanmean((sequence - mean) ** 2)

    # 初始化 ACF 列表
    acf_values = []

    for lag in range(max_lag + 1):
        # 计算协方差
        if lag == 0:
            cov = variance
        else:
            valid_indices = ~np.isnan(sequence[lag:]) & ~np.isnan(sequence[:N - lag])
            y_t = sequence[lag:][valid_indices]
            y_t_lag = sequence[:N - lag][valid_indices]
            cov = np.nanmean((y_t - mean) * (y_t_lag - mean))

        # 计算 ACF 值
        acf_value = cov / variance
        acf_values.append(acf_value)

    significant_acf = np.abs(acf_values[1:]) > (2 / np.sqrt(len(sequence)))
    # 如果ACF值显著，说明序列有较强的自相关性，预测相对容易
    if is_true_majority(significant_acf):
        # print('容易预测')
        return False  # 容易预测
    else:
        # print('不容易预测')
        return True  # 难以预测

def is_true_majority(lst):
    # 计算 True 的个数
    true_count = sum(lst)

    # 计算列表长度
    total_count = len(lst)

    # 判断 True 的个数是否超过一半
    return true_count > total_count / 5



#plot 折线列表 以及突变点
# def draw_plot(linelist,change_points=None):
#
#     colorpointlist=['blue','red','black','yellow','green','purple','pink']
#     y = ['13_7', '13_8','13_9','13_10','13_11','13_12','14_1','14_2','14_3','14_4','14_5','14_6','14_7','14_8','14_9','14_10','14_11','14_12','15_1','15_2','15_3','15_4','15_5','15_6','15_7','15_8','15_9','15_10','15_11','15_12','16_1','16_2','16_3','16_4','16_5','16_6','16_7','16_8','16_9','16_10','16_11','16_12','17_1','17_2','17_3','17_4','17_5','17_6','17_7','17_8','17_9']
#     plt.rcParams['font.size'] = 6
#     for i in range(len(linelist)):
#         plt.plot(y[:len(linelist[i])],linelist[i], marker='', linestyle='-', color=colorpointlist[i], label='Station')
#         plt.xticks(rotation=45)
#
#     if change_points!=None:
#         for k in range(len(linelist)):
#             for pt in change_points:
#                 plt.scatter(pt, linelist[k][pt],s=25, color=colorpointlist[k])
#
#     plt.title('Line Plot of a 1D Array')
#     plt.xlabel('Index')
#     plt.ylabel('Value')
#     # 显示图例
#     plt.legend()
#
#     # 显示图形
#     plt.show()
def draw_plot(linelist,change_points=None):

    colorpointlist=['red','green','black','blue','purple','pink','orange']
    namelist=['week1','week2','week3','week4','week5']
    for i in range(len(linelist)):
        plt.plot(linelist[i], marker='', linestyle='-', color=colorpointlist[i], label=namelist[i])


    if change_points!=None:
        for k in range(len(linelist)):
            for pt in change_points:
                plt.scatter(pt, linelist[k][pt],s=25, color=colorpointlist[k])

    plt.title('Line Plot of a 1D Array')
    plt.xlabel('Index')
    plt.ylabel('Value')
    # 显示图例
    plt.legend()

    # 显示图形
    plt.show()
    
import xml.etree.ElementTree as ET
import os

# def sort_vehicles_by_depart_time(input_xml_path, output_xml_path):
#     """
#     读取XML文件，按照vehicle的depart时间排序，并保存到新文件
    
#     参数:
#     input_xml_path: 输入XML文件路径
#     output_xml_path: 输出XML文件路径
#     """
#     # 解析XML文件
#     tree = ET.parse(input_xml_path)
#     root = tree.getroot()
    
#     # 提取所有vehicle元素
#     vehicles = root.findall('vehicle')
    
#     # 按照depart属性排序（将字符串转换为浮点数进行比较）
#     vehicles.sort(key=lambda v: float(v.get('depart', '0')))
    
    
#     # 清空原有的vehicle元素
#     for vehicle in list(root):
#         if vehicle.tag == 'vehicle':
#             root.remove(vehicle)
    
#     # 按排序后的顺序添加vehicle元素
#     for vehicle in vehicles:
#         root.append(vehicle)
    
#     # 保存到新文件
#     tree.write(output_xml_path, encoding='utf-8', xml_declaration=True)
    

# def sort_vehicles_by_depart_time(input_xml_path, output_xml_path):
#     """
#     读取XML文件，按照vehicle的depart时间排序，并保存到新文件
    
#     参数:
#     input_xml_path: 输入XML文件路径
#     output_xml_path: 输出XML文件路径
#     """
#     # 解析XML文件
#     tree = ET.parse(input_xml_path)
#     root = tree.getroot()
    
#     # 提取所有vehicle元素
#     vehicles = root.findall('vehicle')
    
#     # 按照depart属性排序（将字符串转换为浮点数进行比较）
#     vehicles.sort(key=lambda v: float(v.get('depart', '0')))
#     # 对相同depart时间的车辆进行二次排序
#     # 将vehicles按depart时间分组
#     depart_groups = {}
#     for vehicle in vehicles:
#         depart_time = float(vehicle.get('depart', '0'))
#         if depart_time not in depart_groups:
#             depart_groups[depart_time] = []
#         depart_groups[depart_time].append(vehicle)
    
#     # 对每组内的车辆按ID排序
#     sorted_vehicles = []
#     for depart_time in sorted(depart_groups.keys()):
#         group = depart_groups[depart_time]
#         # 将组内车辆分为原始车辆和后加车辆
#         original = []  # 原始车辆(以't'开头)
#         additional = []  # 后加车辆(以's't'开头)
        
#         for vehicle in group:
#             vehicle_id = vehicle.get('id')
#             if vehicle_id.startswith('s'):
#                 additional.append(vehicle)
#             else:
#                 original.append(vehicle)
                
#         # 对原始车辆和后加车辆分别按ID排序
#         original.sort(key=lambda v: v.get('id'))
#         additional.sort(key=lambda v: v.get('id'))
        
#         # 将排序后的车辆添加到结果列表
#         sorted_vehicles.extend(original)
#         sorted_vehicles.extend(additional)
    
#     # 更新vehicles列表
#     vehicles = sorted_vehicles
#     # 清空原有的vehicle元素
#     for vehicle in list(root):
#         if vehicle.tag == 'vehicle':
#             root.remove(vehicle)
    
#     # 按排序后的顺序添加vehicle元素
#     for vehicle in vehicles:
#         root.append(vehicle)
    
#     # 保存到新文件
#     tree.write(output_xml_path, encoding='utf-8', xml_declaration=True)
def sort_vehicles_by_depart_time(input_xml_path, output_xml_path):
    """
    读取XML文件，按照vehicle的depart时间排序，并保存到新文件
    
    参数:
    input_xml_path: 输入XML文件路径
    output_xml_path: 输出XML文件路径
    """
    # 解析XML文件
    tree = ET.parse(input_xml_path)
    root = tree.getroot()
    
    # 提取所有vehicle元素
    vehicles = root.findall('vehicle')
    
    # 按照depart属性排序（将字符串转换为浮点数进行比较）
    vehicles.sort(key=lambda v: float(v.get('depart', '0')))
    # 对相同depart时间的车辆进行二次排序
    # 将vehicles按depart时间分组
    depart_groups = {}
    for vehicle in vehicles:
        depart_time = float(vehicle.get('depart', '0'))
        if depart_time not in depart_groups:
            depart_groups[depart_time] = []
        depart_groups[depart_time].append(vehicle)
    
    # 对每组内的车辆按ID排序
    sorted_vehicles = []
    for depart_time in sorted(depart_groups.keys()):
        group = depart_groups[depart_time]
        
        # 将组内车辆按基础ID分组
        id_groups = {}
        for vehicle in group:
            vehicle_id = vehicle.get('id')
            # 获取基础ID(去掉前面的's')
            base_id = vehicle_id.lstrip('s')
            if base_id not in id_groups:
                id_groups[base_id] = []
            id_groups[base_id].append(vehicle)
        
        # 对每个基础ID组内的车辆按's'的数量排序
        for base_id in sorted(id_groups.keys()):
            vehicles_same_base = id_groups[base_id]
            vehicles_same_base.sort(key=lambda v: v.get('id').count('s'))
            sorted_vehicles.extend(vehicles_same_base)
    
    # 更新vehicles列表
    vehicles = sorted_vehicles
    # 清空原有的vehicle元素
    for vehicle in list(root):
        if vehicle.tag == 'vehicle':
            root.remove(vehicle)
    count=0
    # 按排序后的顺序添加vehicle元素
    for vehicle in vehicles:
        root.append(vehicle)
        count+=1
    
    # 保存到新文件
    tree.write(output_xml_path, encoding='utf-8', xml_declaration=True)
    return count


def find_basd_casepair(rmse_per,complete_truths,h_index1,h_index2,name_id_list):
    """The data loader that extracts and processes data from a :obj:`DataSet` object.

       Args:
           rmse_per (dict): 每个站点不同时刻的rmse
           complete_truths (array):整体序列
           h_index1,h_index2:历史预测序列在complete_truths 数组中的索引

       """
    badcase = []
    for i in rmse_per.keys():
        iop_sequence = complete_truths[i, h_index1:h_index2]
        import pdb
        pdb.set_trace()
        print((rmse_per[i][:]))

        if max(rmse_per[i][:]) > 200 and calculate_acf(iop_sequence, 72):
            # maxindex = np.argmax(rmse_per[i][:])
            badcase.append(name_id_list[i])

            count = len([num for num in rmse_per[i][:] if num > 200])
            # if count>

    return  badcase




def peak_detect(time_series, threshold1=1.2,threshold2=1.1):

    n = len(time_series)
    change_points = []

    change_points.append(0)
    change_points.append(0)
    change_points.append(0)
    change_points.append(0)
    for i in range(4,len(time_series)):

        # 计算当前时刻加上之前两个时刻的平均值
        curr_avg=np.mean([ time_series[k] for k in range(i-2,i+1) if time_series[k]!=0])
        pre_avg=np.mean([ time_series[k] for k in range(i-3,i) if time_series[k]!=0])

        if time_series[i]>threshold1*time_series[i-1] and curr_avg>threshold2*pre_avg and pre_avg>0.0001 :
            change_points.append(1)
        else:
            change_points.append(0)

    return change_points



#plot 折线列表 以及突变点
def draw_Mullines_in_one_picture(linelist,change_points=None):
    print(len(linelist))

    colorpointlist=['blue','red','black','orange','green','purple','pink']

    for i in range(len(linelist)):
        plt.plot(linelist[i], marker='', linestyle='-', color=colorpointlist[i],label='Day{}'.format(i+1),alpha=0.5)
        if change_points!=None and change_points[i] != None:
            print(change_points[i])
            for pt in change_points[i]:
                plt.scatter(pt, linelist[i][pt], s=25, color=colorpointlist[i])

    plt.title('Line Plot of a 1D Array')
    plt.xlabel('Timeslot')
    plt.ylabel('Waiting Time')
    # 显示图例
    plt.legend()

    # 显示图形
    plt.show()


def transform1_line(line,changpoint):
    newline=[]
    count=0
    for i in range(len(line)):
        if i in changpoint and  count<=20:
            count+=1
            newline.append(random.uniform(1.5, 2.5))
        else:
            newline.append(0)
    return  newline


def transform_pointlist(list,front_n):
    count=0
    newlist=[]
    calculator=0
    for i in list:
        calculator+=1
        if i!=0 and count<=front_n:
            newlist.append(1)
            count += 1
        else:
            newlist.append(0)
    return  newlist


#产生训练数据
def generate_data(oringin_data_file,station_name,day_range,thres1,thres2,thres3,thres4,dir_='0_1_label_pkl',timefitness=72):
    trainlist=[]
    labellist=[]
    with open('F:\TianJinData\新建文件夹\item_id_list_Tianjin.pkl', 'rb') as file:
        name_id_list = pickle.load(file)
    days = 74
    peak_time = 25
    if timefitness == 144:
        days = 45
        peak_time = 50
    complete_truths = np.load(oringin_data_file)
    complete_truths = np.transpose(complete_truths)

    complete_truths=np.nan_to_num(complete_truths)

    newarray=complete_truths.reshape((28970,days,timefitness))
    if day_range[0]==-14:
        newarray=newarray[:,-14:]
    else:
        newarray = newarray[:, day_range[0]:day_range[1]]
    a, b, c = station_name.split('_')
    c = int(c)
    previous_numlist=[]
    for numstop in range(-2,3):
        if numstop==0:
            continue
        compare_pair1 = a + '_' + b + '_' + str(c + numstop)
        if compare_pair1 not in name_id_list:
            print('没有上下游站点错误')
            return
        previous_numlist.append(name_id_list.index(compare_pair1))

    previous_station_data=newarray[previous_numlist,:]
    Object_station_data=newarray[name_id_list.index(station_name),:]

    previous_station_changedata=[]
    Object_station_changedata=[]



    # 临近站点处理
    count_thres=0
    for j in previous_station_data:
        iop=[]
        for k in j:
            k=fill_zeros(k)

            changepoint = peak_detect(k,0,thres2)
            pointlist=transform_pointlist(changepoint,100)
            iop.append(pointlist.copy())
        count_thres+=1
        previous_station_changedata.append(iop.copy())
    closepair=np.array(previous_station_changedata).reshape((4,(abs(day_range[0])-abs(day_range[1]))*timefitness))


    # pos = bisect.bisect_left(result, 0.22)
    # print(pos/len(result))
    # import pdb
    # pdb.set_trace()
    # print(result)
    #目标站点处理
    for p in Object_station_data:
        p = fill_zeros(p)

        changepoint = peak_detect(p,0,thres4)
        pointlist=transform_pointlist(changepoint,100)
        Object_station_changedata.append(pointlist.copy())
    Objectpair=np.array(Object_station_changedata).reshape(((abs(day_range[0])-abs(day_range[1]))*timefitness))


    count=0
    for y in range(4,(abs(day_range[0])-abs(day_range[1]))*timefitness):
        label=Objectpair[y]
        if y % timefitness > 35 or y % timefitness < 17:
            continue
        if label == 1:
            count+=1
    cal=0
 
    if count==0:
        return [],[]
    for t in range(14,(abs(day_range[0])-abs(day_range[1]))*timefitness):

        # if t% timefitness>35 or t% timefitness< 17 :
        #     continue

        train_i=closepair[:,t-4:t]
        label=Objectpair[t]
        # trainlist.append(train_i.reshape(-1))
        # labellist.append(label)
        if label == 1:
            trainlist.append(train_i.reshape(-1))
            labellist.append(label)
            for r in range(-4,0):
                if Objectpair[t+r]==0:
                    trainlist.append(closepair[:,t+r-4:t+r].reshape(-1))
                    labellist.append(Objectpair[t+r])
                    break


        #
        #
        # if label == 0 :
        #     # if cal<count:
        #     #     cal += 1
        #     trainlist.append(train_i.reshape(-1))
        #     labellist.append(label)
        #     # else:
        #     #     continue

    return np.array(trainlist),np.array(labellist)


#
# # 给14天的测试数据打上label
# def generate_testdata_label(oringin_data_file,station_name,day_range,thres1,thres2,thres3,thres4,dir_='0_1_label_pkl',timefitness=72):
#     with open('F:\TianJinData\新建文件夹\item_id_list_Tianjin.pkl', 'rb') as file:
#         name_id_list = pickle.load(file)
#
#     complete_truths = np.load(oringin_data_file)
#     complete_truths = np.transpose(complete_truths)
#
#     complete_truths=np.nan_to_num(complete_truths)
#     days=74
#     peak_time=25
#     if timefitness==144:
#         days=45
#         peak_time=50
#
#
#     newarray=complete_truths.reshape((28970,days,timefitness))
#     if day_range[0]==-14:
#         newarray=newarray[:,-14:]
#     else:
#         newarray = newarray[:, day_range[0]:day_range[1]]
#     a, b, c = station_name.split('_')
#     c = int(c)
#     previous_numlist=[]
#     for numstop in range(-2,3):
#         # if numstop==0:
#         #     continue
#         compare_pair1 = a + '_' + b + '_' + str(c + numstop)
#         if compare_pair1 not in name_id_list:
#             print('没有上下游站点错误')
#             return
#         previous_numlist.append(name_id_list.index(compare_pair1))
#
#     previous_station_data=newarray[previous_numlist,:]
#
#     previous_station_changedata=[]
#
#     list_thres_all=[]
#     for w in previous_station_data:
#         list_thres=[]
#         for k in w:
#             k=fill_zeros(k)
#             list_q = calculate_changes(k[0:peak_time])
#             list_thres = list_thres + list_q
#         result = sorted(list_thres)
#         index_=int(len(result) * 0.7)
#         list_thres_all.append(np.mean(result[ index_- 1:index_ + 1]))
#
#     # 临近站点处理
#     count_thres=0
#     for j in previous_station_data:
#         iop=[]
#         for k in j:
#             k=fill_zeros(k)
#             changepoint = peak_detect(k,0,list_thres_all[count_thres]+1)
#             pointlist=transform_pointlist(changepoint,200)
#             iop.append(pointlist.copy())
#         count_thres+=1
#         previous_station_changedata.append(iop.copy())
#     closepair=np.array(previous_station_changedata).reshape((5,(abs(day_range[0])-abs(day_range[1]))*timefitness))
#
#     with open('{}/closepairlabel_{}.pkl'.format(dir_,station_name), 'wb') as f:
#         pickle.dump(closepair,f)
#     return 0

def find_peak_rmse(rmse_svm,rmse_hm):
    list1 = rmse_svm
    list2 = rmse_hm
    # 存储不一样的值的差值
    differences = []
    # 遍历两个列表并比较元素
    cal=0
    for a, b in zip(list1, list2):
        if a != b:
            differences.append(b-a)
        # if b-a<-200:
        #     print(cal)
        #     import pdb
        #     pdb.set_trace()
        cal+=1
    return differences




# 给14天的测试数据打上label
def generate_testdata_label(oringin_data_file,station_name,day_range,thres1,thres2,thres3,thres4,dir_='0_1_label_pkl',timefitness=72):
    with open('F:\TianJinData\新建文件夹\item_id_list_Tianjin.pkl', 'rb') as file:
        name_id_list = pickle.load(file)

    complete_truths = np.load(oringin_data_file)
    complete_truths = np.transpose(complete_truths)

    complete_truths=np.nan_to_num(complete_truths)
    days=74
    peak_time=25
    if timefitness==144:
        days=45
        peak_time=50


    newarray=complete_truths.reshape((28970,days,timefitness))
    if day_range[0]==-14:
        newarray=newarray[:,-14:]
    else:
        newarray = newarray[:, day_range[0]:day_range[1]]
    a, b, c = station_name.split('_')
    c = int(c)
    previous_numlist=[]
    for numstop in range(-2,3):
        if numstop==0:
            continue
        compare_pair1 = a + '_' + b + '_' + str(c + numstop)
        if compare_pair1 not in name_id_list:
            print('没有上下游站点错误')
            return
        previous_numlist.append(name_id_list.index(compare_pair1))

    previous_station_data=newarray[previous_numlist,:]

    previous_station_changedata=[]
    Object_station_data = newarray[name_id_list.index(station_name), :]






    # 临近站点处理
    for j in previous_station_data:
        iop=[]
        for k in j:
            k=fill_zeros(k)
            changepoint = peak_detect(k,0,thres2)
            pointlist=transform_pointlist(changepoint,200)
            iop.append(pointlist.copy())
        previous_station_changedata.append(iop.copy())

    #目标节点 处理
    Object_station_changedata = []
    for p in Object_station_data:
        p = fill_zeros(p)
        changepoint = peak_detect(p,0,thres4)
        pointlist=transform_pointlist(changepoint,100)
        Object_station_changedata.append(pointlist.copy())
    Objectpair=np.array(Object_station_changedata).reshape(((abs(day_range[0])-abs(day_range[1]))*timefitness))




    closepair=np.array(previous_station_changedata).reshape((4,(abs(day_range[0])-abs(day_range[1]))*timefitness))
    Objectpair = np.expand_dims(Objectpair, axis=0)
    closepair=np.concatenate((closepair, Objectpair), axis=0)

    with open('{}/closepairlabel_{}.pkl'.format(dir_,station_name), 'wb') as f:
        pickle.dump(closepair,f)
    return 0

def whether_exist_break(array,threshold):

    return np.count_nonzero(array == 1)>=threshold



def dumpfile(data,filename):
    with open(filename, 'wb') as file:
        # 使用 pickle.dump 将数据序列化并保存到文件
        pickle.dump(data,file)
    return 0

def loadfile(filename):
    with open(filename, 'rb') as file:
        # 使用 pickle.dump 将数据序列化并保存到文件
        data=pickle.load(file)
    return data


def plot_new(original_array):

    x = np.arange(len(original_array))
    y = []
    for i, num in enumerate(original_array):
        if i == 0:
            y.append(num)
        else:
            if num != 0:
                y.append(num)
            else:
                y.append(y[-1])
    return  x,y


#计算奖惩制度中最高分的算法
def find_highest_score_threshold(reward_scores, penalty_scores, threshold_values):
    max_ratio = -1  # 初始化最大比值为-1，因为比值不能为负
    best_threshold = None  # 初始化最佳阈值为None
    for i in range(len(reward_scores)):
        reward = reward_scores[i]
        penalty = penalty_scores[i]
        # 只有当奖励和惩罚都不为零时才计算比值
        if reward != 0 and penalty != 0:
            ratio = reward / penalty
            if ratio > max_ratio:
                max_ratio = ratio
                best_threshold = threshold_values[i]
    return best_threshold,max_ratio

def find_highest_score_when_punishZero(reward_scores, penalty_scores, threshold_values):
    max_ratio = -1  # 初始化最大比值为-1，因为比值不能为负
    best_threshold = None  # 初始化最佳阈值为None
    for i in range(len(reward_scores)):
        reward = reward_scores[i]
        penalty = penalty_scores[i]
        # 只有当奖励和惩罚都不为零时才计算比值
        if  penalty == 0:
            if reward > max_ratio:
                max_ratio = reward
                best_threshold = threshold_values[i]
    return best_threshold,max_ratio
#聚类分辨拥堵时间和非拥堵时间
def bias_asbreak2(congestion_times):
    X = np.array(congestion_times).reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    labels = kmeans.labels_
    cluster0_indices = np.where(labels == 0)[0]
    cluster1_indices = np.where(labels == 1)[0]
    cluster0_mean = np.mean(X[cluster0_indices])
    cluster1_mean = np.mean(X[cluster1_indices])
    # print(X[cluster0_indices])
    # print(X[cluster1_indices])
    if cluster0_mean > cluster1_mean:
        # print(X[cluster0_indices])
        return cluster0_mean,X[cluster0_indices]
    else:
        # print(X[cluster1_indices])
        return cluster1_mean,X[cluster1_indices]


def is_stable(list_,test):
    if test<list_[0]:
        return  True
    else:
        return False



def bias_asbreak(value):
    return value+value*0.3

#算一个站点总的rmse
def calculate_rmse_per_station(pred, true):
    rmse_per_station = {}
    pre_mask,true_mask=pred,true
    meanrmse= np.sqrt(np.mean((pre_mask - true_mask) ** 2))

    # 遍历每个站点
    for station in range(pred.shape[0]):
        # 获取当前站点的预测值、真实值和掩码
        pred_station = pred[station]
        true_station = true[station]
        # mask_station = mask[station]

        # 筛选出有效元素
        # valid_indices = np.where(mask_station)
        pred_valid = pred_station
        true_valid = true_station

        # 计算差值平方
        diff_squared = (pred_valid - true_valid) ** 2
        diff_squared=np.nan_to_num(diff_squared)

        diff = diff_squared[diff_squared != 0]
        diff=np.sum(diff)/len(diff)
        # 计算RMSE
        rmse = np.sqrt(diff)

        # 存储当前站点的RMSE
        rmse_per_station[station] = rmse


    return rmse_per_station,meanrmse




#每个时刻的rmse都计算
def calculate_rmse_per_station_per_slice(pred, true,station):
    rmse_per_station = {}
    pre_mask,true_mask=pred,true
    meanrmse= np.sqrt(np.mean((pre_mask - true_mask) ** 2))

    # 遍历每个站点

    pred_station = pred[station]
    true_station = true[station]
    # mask_station = mask[station]

    # 筛选出有效元素
    # valid_indices = np.where(mask_station)
    pred_valid = pred_station
    true_valid = true_station

    # 计算差值平方
    diff_squared = (pred_valid - true_valid) ** 2
    diff_squared=np.nan_to_num(diff_squared)

    # diff = diff_squared[diff_squared != 0]
    #diff=np.sum(diff)/len(diff)
    # 计算RMSE
    rmse = np.sqrt(diff_squared)

    # 存储当前站点的RMSE
    rmse_per_station[station] = rmse


    return rmse_per_station,meanrmse



#rmse聚合
def calculate_rmse_per_station_aggrevate(pred, true,station):
    rmse_per_station = {}
    pre_mask,true_mask=pred,true
    meanrmse= np.sqrt(np.mean((pre_mask - true_mask) ** 2))


    # 获取当前站点的预测值、真实值和掩码
    pred_station = pred[station]
    true_station = true[station]
    pred_valid = pred_station
    true_valid = true_station

    # 计算差值平方
    diff_squared = (pred_valid - true_valid) ** 2


    #diff=np.sum(diff)/len(diff)
    # 计算RMSE
    rmse = diff_squared

    # 存储当前站点的RMSE
    rmse_per_station[station] = rmse


    return rmse_per_station

#数据清洗，对缺失值进行处理
def fill_zeros(sequence):
    # Start with the first element
    previous_value = sequence[0]
    filled_sequence = []

    for num in sequence:
        if num == 0:
            # Replace 0 with the previous non-zero value
            filled_sequence.append(previous_value)
        else:
            # Update previous_value and append the current number
            filled_sequence.append(num)
            previous_value = num

    return filled_sequence

#
def merge_data(data, dataType,MergeWay,MergeIndex):
    if MergeWay == "sum":
        func = np.sum
    elif MergeWay == "average":
        func = np.mean
    elif MergeWay == "max":
        func = np.max
    else:
        raise ValueError("Parameter MerWay should be sum or average")
    if data.shape[0] % MergeIndex is not 0:
        raise ValueError("time_slots % MergeIndex should be zero")

    if dataType.lower() == "node":
        new = np.zeros((data.shape[0] // MergeIndex, data.shape[1]), dtype=np.float32)
        for new_ind, ind in enumerate(range(0, data.shape[0], MergeIndex)):
            new[new_ind, :] = func(data[ind:ind + MergeIndex, :], axis=0)
    elif dataType.lower() == "grid":
        new = np.zeros((data.shape[0] // MergeIndex, data.shape[1], data.shape[2]), dtype=np.float32)
        for new_ind, ind in enumerate(range(0, data.shape[0], MergeIndex)):
            new[new_ind, :, :] = func(data[ind:ind + MergeIndex, :, :], axis=0)
    return new




def loadxmlfile(file_path):
    """
    读取XML文件并返回其内容
    
    参数:
    file_path: XML文件的路径
    
    返回:
    解析后的XML内容
    """
    import xml.etree.ElementTree as ET
    
    try:
        # 解析XML文件
        tree = ET.parse(file_path)
        # 获取根元素
        root = tree.getroot()
        return root
    except ET.ParseError as e:
        print(f"XML解析错误: {e}")
        return None
    except FileNotFoundError:
        print(f"文件未找到: {file_path}")
        return None
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return None
