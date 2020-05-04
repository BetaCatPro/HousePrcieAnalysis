# PricesDataAnalysis
本项目使用jupyter notebook开发，主要目的是分析成都二手房房价，[项目地址](https://github.com/BetaCatPro/HousePrcieAnalysis)。

**数据**：爬取二手房交易网站近期数据，成都各个区域交易热度较高的房屋信息。
[爬虫项目地址](https://github.com/BetaCatPro/Joint-spiders)
**目标**：分析成都各区域二手房市场走势，了解各区域交易情况，建立简单机器学习模型预测房价，及进行聚类分析各房源具体分布情况。

技术点：
- Pandas
- Numpy
- sklearn
- matplotlib

基本流程：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200501204631344.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2J5NjY3MTcxNQ==,size_16,color_FFFFFF,t_70#pic_center)
## 一. 数据采集
数据抓取项目为另一个项目：[项目地址](https://github.com/BetaCatPro/Joint-spiders)，主要抓取房源的特征有：
| 字段名称          | 字段含义     |
| :---------------- | :----------- |
| title             | 房源名称     |
| price             | 房源总价     |
| unit_price        | 房源单价     |
| community_name    | 所在小区名字 |
| region            | 所在行政区划 |
| type              | 户型         |
| construction_area | 建筑面积     |
| orientation       | 房屋朝向     |
| decoration        | 装修情况     |
| floor             | 楼层         |
| elevator          | 电梯情况     |
| purposes          | 房屋用途     |
| release_date      | 挂牌时间     |
| image_urls        | 房源图片     |
| from_url          | 房源来源     |
| house_structure   | 建筑结构     |

爬取完成后导入Excel文件
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200501210110808.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2J5NjY3MTcxNQ==,size_16,color_FFFFFF,t_70#pic_center)
另外，考虑到后边将进行地图展示，所以还需增加地理坐标信息：经纬度，这部分将在数据清洗后进行。

## 二. 数据清洗
**1. 原始数据检视**

基于我爬虫项目的存储策略，我将个区划的结果分别存储到了不同的文件，所以要进行文件合并操作。
首先读取文件列表，然后对循环文件列表，进行合并任务：

```python
datas = []
for file in res:
    filename = file.replace('.csv','')
    try:
        data = pd.read_csv(file)
        datas.append(data)
    except:
        print('%s暂无数据'%filename)
        
# 得到所有合并数据
result = pd.concat(datas)
```
这里我们就得到了总体数据集，使用`result.info()`及`result.shape`查看基本信息：
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020050121262299.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2J5NjY3MTcxNQ==,size_16,color_FFFFFF,t_70#pic_center)
通过以上代码，可以看出训练集数据总共有110556条房屋记录，总共有16列数据，仔细检查数据，可以发现存在很多的缺失值。

**2. 数据的探索性可视化分析**

数据里面有的值大，有的值小，有的列还有缺失值等等，使用pandas_profiling模块工具一键生成探索性数据分析报告,快速查看这些数据的分布 。

```python
ppf.ProfileReport(df_train)
```


![在这里插入图片描述](https://img-blog.csdnimg.cn/20200501213640950.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2J5NjY3MTcxNQ==,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200501213640993.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2J5NjY3MTcxNQ==,size_16,color_FFFFFF,t_70)

![在这里插入图片描述](https://img-blog.csdnimg.cn/202005012136413.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2J5NjY3MTcxNQ==,size_16,color_FFFFFF,t_70)
通过以上数据探索性分析报告可以看出数据集的基本信息、哪些特征属性的缺失值和0元素的占比情况、各特征变量的分布情况以及相关性等等。

**3. 数据清洗**

3.1 去重
查看重复值数量：

```python
result.duplicated().value_counts()
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200504214925515.png#pic_center)
False表示未重复的数目，True表示重复数目。通过`drop_duplicates`方法去除数据集中所有重复值：

```python
res = result.drop_duplicates(subset=None,keep='first',inplace=False)
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020050421505423.png#pic_center)
3.2 检测与处理缺失值
查看缺失值统计结果：
```python
res.isnull().sum()
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200504215253350.png#pic_center)
上面可以看出房屋朝向(orientation)，装修情况(decoration)，建筑结构(house_structure)存在大量缺失值。关于缺失值处理有很多处理方法，比如直接删除，使用随即森林法填充等，这里我们使用特定数据进行填充。定义房屋朝向列表`['东','南','西','北','东南','西南','东北','西北']`，装修情况列表`['简装','精装','毛坯','其他']`，建筑结构列表 `['钢混结构','钢结构','混合结构','框架结构','未知','砖混结构','砖木结构']`。用这里的值进行随机填充。

```python
res1 = res.copy()
orientations = ['东','南','西','北','东南','西南','东北','西北']
decorations = ['简装','精装','毛坯','其他']
house_structures = ['钢混结构','钢结构','混合结构','框架结构','未知','砖混结构','砖木结构']
res1['orientation'].fillna(random.choice(orientations),inplace=True)
res1['decoration'].fillna(random.choice(decorations),inplace=True)
res1['house_structure'].fillna(random.choice(house_structures),inplace=True)
```
3.3 检测异常值
这之前先将面积特征转换为浮点数类型：`res1['construction_area'] = res1['construction_area'].str.replace('㎡','').astype("float")`，去掉'㎡'。此时查看数据集描述信息，包括最小值，下四分位数，均值，上四分位数，最大值，方差，数量信息。

```python
res1.describe()
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200504220628197.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2J5NjY3MTcxNQ==,size_16,color_FFFFFF,t_70#pic_center)
这里可以看到存在房价为0的数据，以及不合理的面积数值，稍后做相关处理。
接下来检查面积与价格之间的关系图：

```python
plt.figure(figsize=(16,15)) 
plt.subplot(221)
plt.scatter(res1["construction_area"], res1["price"])
plt.xlabel('建筑面积',fontsize=15)
plt.ylabel('总价',fontsize=15)
 
plt.subplot(222)
plt.scatter(res1["construction_area"], res1["unit_price"])
plt.xlabel('建筑面积',fontsize=15)
plt.ylabel('单价',fontsize=15)
plt.show()
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200504220903234.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2J5NjY3MTcxNQ==,size_16,color_FFFFFF,t_70#pic_center)
可以明显观察到存在异常情况。最后观察房价的箱线图：

```python
plt.figure(figsize=(16,8)) 
plt.subplot(1,2,1)
plt.boxplot(res1["price"])
plt.ylabel('总价',fontsize=15)
plt.subplot(1,2,2)
plt.boxplot(res1["unit_price"])
plt.ylabel('单价',fontsize=15)
plt.show()
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200504221105144.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2J5NjY3MTcxNQ==,size_16,color_FFFFFF,t_70#pic_center)
3.4 处理异常值
通过上面观察分析，房价个面积都存在异常情况，对其分别处理。
首先处理离群值和有失一般性值，比如上图中的面积：

```python
res1.drop(res1[res1['construction_area']>1000].index,inplace=True)
```
处理price和unit_price为 0 的数据
```python
# 查看相关数据
print(res1[res1['unit_price']==0])
print(res1[res1['unit_price']==0]['community_name'])
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200504221443528.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2J5NjY3MTcxNQ==,size_16,color_FFFFFF,t_70#pic_center)
由于数据量过小，所以直接删除`res1.drop(res1[res1['price']==0].index,inplace=True)`,但是如果对于存在一定数量相关值时，不能直接删除，这样会影响数据。这里可以采用一种替换值法：获取到每条数据对应的小区的均价，用这个均价来填充房源单价，面积同样采用这个方法，最后房屋总价通过计算单价和面值积获得。当然也可用机器学习算法建模获取与待处理目标最相近的房源的数据来填充该处理目标。处理后的面积散点图：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200504222159595.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2J5NjY3MTcxNQ==,size_16,color_FFFFFF,t_70#pic_center)
3.5 其他标准
绘制出装修情况，建筑结构，房屋用途，房屋面积与房价的散点图：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200504222408539.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2J5NjY3MTcxNQ==,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200504222408517.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2J5NjY3MTcxNQ==,size_16,color_FFFFFF,t_70)
通过散点图可以观察到哪些是异常值点，例如：construction_area与price的关系图中，有几个离群的 construction_area值很高的数据，可以推测出现这种情况的原因。或许他们代表了相当高级地区，也就解释了高价。 这些点很明显不能代表典型样例，所以我们将它们定义为异常值并删除。
同理，对于其他特征存在的不合理的离群点，在这里也考虑将其删除。

```python
res1.drop(res1[(res1['decoration']=='其他') & (res1['price']>6000)].index,inplace=True)
res1.drop(res1[(res1['house_structure']=='钢混结构') & (res1['price']>7000)].index,inplace=True)
res1.drop(res1[(res1['house_structure']=='框架结构') & (res1['price']>6000)].index,inplace=True)
res1.drop(res1[(res1['house_structure']=='未知结构') & (res1['price']>4000)].index,inplace=True)
res1.drop(res1[(res1['purposes']=='普通住宅') & (res1['price']>6000)].index,inplace=True)
res1.drop(res1[(res1['construction_area']>700) & (res1['price']<300)].index,inplace=True)
res1.drop(res1[(res1['construction_area']<600) & (res1['price']>4000)].index,inplace=True)
complete_data = res1.copy()
```
至此，简单的数据清洗就完成了，接下俩要完成的是地理坐标转换功能，后续使用聚类进行地图应用展示的时候需要用到地理坐标，所以我们要将每房源的地理位置解析出来，合并到数据中。

## 三. 逆地址解析
为数据集添加索引`# complete_data['id'] = range(len(complete_data))`使得其连续。

这里我们使用高德地图解析进行具体地址转换为经纬度操作，使用高德地图webapi前，需要申请到高德地图的key(百度地图为ak)，才能使用相关接口，百度地图同理。
注：
1. 地理编码/逆地理编码 API 是通过 HTTP/HTTPS 协议访问远程服务的接口，提供结构化地址与经纬度之间的相互转化的能力。
2. 此处选择高德地图是因为我在使用百度地图的webapi时频繁断开链接，导致解析失败，所以选择了高德地图。但是后面还要进行一次高德地图坐标转换百度地图坐标，之所以这样做是因为我的另一个项目使用的是百度地图做的地图可视化，所以如果这里使用百度地图服务的话，就能省去后面的坐标转换步骤。

使用：
	- 申请Web服务API类型Key
	- 参考接口参数文档发起HTTP/HTTPS请求，第一步申请的 Key 需作为必填参数一同发送
	- 接收请求返回的数据（JSON或XML格式），参考返回参数文档解析数据。
地理编码 API 服务地址：`https://restapi.amap.com/v3/geocode/geo?parameters`
请求方式：GET

具体参数及说明见[高德地图开发者文档](https://lbs.amap.com/api/webservice/guide/api/georegeo)。


**1.  定义转换函数**
```python
def getlnglat_gaode(address):
    address = quote(address)
    # api
    url_base = "http://restapi.amap.com/v3/geocode/geo"
    # 返回数据格式
    output = "json"
    # key
    key = "5d297ac38ce0db596ad9656b13fa9b08"
    url = url_base + '?' + 'address=' + address  + '&output=' + output + '&key=' + key
    
    lat = 0.0
    lng = 0.0
    res = requests.get(url)
    temp = json.loads(res.text)
    location = temp['geocodes'][0]['location'].split(',')
    if temp["info"] == 'OK':
        lat = location[1]
        lng = location[0]
    # 返回解析好的坐标
    return lat,lng
```
测试：`lat,lang = getlnglat_gaode('四川省成都市新津金秋乐园一期')`
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200504224930636.png#pic_center)
**2. 处理全部数据**

2.1 定义基本数据结构：

```python
# 索引
idint = []
# 小区名
community_names = []
# 经纬度
lats = []
lngs = []
# 完整地址
address = ''
# 格式化数据
lat_lng_data = {"id":idint,"community_name":community_names,"lat":lats,"lng":lngs}
```
2.2 生成经纬度信息，这里我们的数据保存策略是每两千条存储到一个CSV文件中，以免断开链接后数据丢失的问题：

```python
for idi,community_name,region in zip(list(complete_data["id"]),list(complete_data["community_name"]),list(complete_data["region"])):
	# 获取小区名并生成完整地址
    community_name = str(community_name)
    region = re.sub(r"\[|\]|'","",region).split(',')
    if len(region)>=2:
        if region[0] != region[1]:
            address = "成都市"+region[0]+region[1]+community_name
        else:
            address = "成都市"+region[0]+community_name
    else:
        address = "成都市"+region[0]+community_name
#     print(address)
#     print('*'*20)
	# 解析地址
    lat,lng = getlnglat_gaode(address)
    if lat != 0 or lng !=0:
        idint.append(idi)
        community_names.append(community_name)
        lats.append(lat)
        lngs.append(lng)
        print(idi,lat,lng)
    # 分段存储
    if idi>0 and idi%2000==0:
        df_latlng = pd.DataFrame(lat_lng_data)
        df_latlng.to_csv("./cleandata/latlng"+str(idi)+".csv",encoding='gbk')
        idint = []
        community_names = []
        lats = []
        lngs = []
        address = ''
        lat_lng_data = {"id":idint,"community_name":community_names,"lat":lats,"lng":lngs}
```
过程截取：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200504225728572.png#pic_center)
由于数据不能整数2000，所以还会遗留一部分数据，接下来将这部分数据存储：`df_latlng = pd.DataFrame(lat_lng_data)
df_latlng.to_csv("./cleandata/latlng100983.csv")`

**3. 合并所有坐标文件**

将所有坐标文件合并在一起，方便合并到房源数据集中。

```python
position_name = os.listdir('./cleandata/')
res = [position for position in position_name]

datas = []
for file in res:
    filename = file.replace('.csv','')
    file = './cleandata/'+file
    try:
        data = pd.read_csv(file,encoding='gbk')
        datas.append(data)
    except:
        print('%s暂无数据'%filename)

# 得到所有合并数据
position_result = pd.concat(datas)
position_result.to_csv('./cleandata/lnglat.csv')
```
**4. 合并得到最终数据**

将房源数据集和做坐标数据集按Id合并，保证数据对应的一致性，由于前边做坐标转换时是根据id来存数据的，所以不存在数据对应出错的问题。

```python
del position_result["community_name"]
df_merge = pd.merge(complete_data,position_result,on="id")
df_merge.to_csv('./housedata/fin_house.csv')
```
注：这里的最终数据`fin_house.csv`中的坐标是遵循高德地图坐标，如果是做高德地图应用的话，就可直接使用了，但我是采用的百度地图，所以我还要在进行高德地图和百度地图的坐标转换，以及坐标纠正，不需要这一步的同学可以跳过。

## 四. 高德坐标转百度坐标
**1. 定义转换函数，实现坐标对接：**

相关参数详情见百度地图开发者文档。
```python
def parse2lnglat(lng,lat):
	# 百度api
    url_base = "http://api.map.baidu.com/geoconv/v1/?coords="
    # 返回数据格式
    output = "json"
    ak = "Qmz0VMtKw3uAI2GWClu9Q6iCnP2j2uH2"
    url = url_base + str(lng) +','+ str(lat) + '&output=' + output + '&ak=' + ak

    res = requests.get(url)
    temp = json.loads(res.text)
    lng=0
    lat=0
    if temp['status']==0:
        lng = temp['result'][0]['x']
        lat = temp['result'][0]['y']
    return lat,lng
```

测试：`lat,lng = parse2lnglat(104.006705,30.577101)`
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200504231245951.png#pic_center)
**2. 生成经纬度信息**

这一步和上面解析地址类似
```python
# 生成经纬度信息
idint = []
community_names = []
lats = []
lngs = []
lat_lng_data = {"id":idint,"community_name":community_names,"lat":lats,"lng":lngs}

for idi,lat,lng,community_name in zip(list(pre_location["id"]),list(pre_location["lat"]),list(pre_location["lng"]),list(pre_location["community_name"])):
    lat = str(lat)
    lng = str(lng)
    community_name = str(community_name)
    lat,lng = parse2lnglat(lng,lat)
    if lat != 0 or lng !=0:
        idint.append(idi)
        community_names.append(community_name)
        lats.append(lat)
        lngs.append(lng)
        print(idi,lat,lng)
    if idi>0 and idi%2000==0:
        df_latlng = pd.DataFrame(lat_lng_data)
        df_latlng.to_csv("./cleandata/updateposition/latlng"+str(idi)+".csv",encoding='gbk')
        idint = []
        community_names = []
        lats = []
        lngs = []
        address = ''
        lat_lng_data = {"id":idint,"community_name":community_names,"lat":lats,"lng":lngs}
```
处理剩下的数据:
`df_latlng = pd.DataFrame(lat_lng_data)
df_latlng.to_csv("./cleandata/updateposition/latlng100983.csv",encoding='gbk')`

**3. 合并数据集**

```python
position_name = os.listdir('./cleandata/updateposition/')
res = [position for position in position_name]

datas = []
for file in res:
    filename = file.replace('.csv','')
    file = './cleandata/updateposition/'+file
    try:
        data = pd.read_csv(file,encoding='gbk')
        datas.append(data)
    except:
        print('%s暂无数据'%filename)

# 得到所有合并数据
position_result = pd.concat(datas)
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200504231631965.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2J5NjY3MTcxNQ==,size_16,color_FFFFFF,t_70#pic_center)
删除Unnaemd列，并保存为CSV文件:
`
del position_result['Unnamed: 0']
position_result.to_csv('./cleandata/updateposition/lnglat.csv')
`
这时的`fin_house2.csv`为百度坐标格式的文件，可使用到百度地图应用中去。

## 五. 特征工程

