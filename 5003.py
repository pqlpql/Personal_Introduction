import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
def load_data_label(filename):
    '''
    Load data with label specified
    '''
    data = []
    with open(filename) as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip().split()
            data_line = [float(i) for i in line]
            data.append(data_line)
    return data
def addKey(x, max_x, max_y, buffer):
    mid_x = max_x / 2
    mid_y = max_y / 2
    cur = x
    l = []
    if cur[0] <=mid_x + buffer and cur[1]<=mid_y + buffer:
        l.append((0,x))
    if cur[0] > mid_x - buffer and  cur[1] <=mid_y + buffer:
        l.append((1,x))
    if  cur[0] <=mid_x + buffer and cur[1]>mid_y-buffer:
        l.append((2,x))
    if cur[0] > mid_x - buffer and cur[1] >mid_y-buffer:
        l.append((3,x))
    return l

def reduce(x,y):
    if isinstance(x[0], np.float64) and isinstance(y[0], np.float64):
        return np.concatenate((x,y), axis=0).reshape((2,2))
    elif isinstance(x[0], np.ndarray) and isinstance(y[0], np.float64):
        return np.concatenate((y.reshape((1,2)),x), axis=0)
    elif isinstance(x[0], np.float64) and isinstance(y[0], np.ndarray):
        return np.concatenate((x.reshape((1,2)),y), axis=0)
    else:
        return np.concatenate((x,y), axis=0)
def topd(x):
    return (x[0], pd.DataFrame(x[1], columns = ['feature1','feature2']))
def dataProcessing(data):
    rdd = sc.parallelize(train_data,10)
    max_x = max(data[:,0])
    max_y = max(data[:,1])
    rdd = rdd.map(lambda x: addKey(x, max_x, max_y, 3)).reduce(lambda x, y: x+y)
    rdd = sc.parallelize(rdd).reduceByKey(lambda x, y: reduce(x,y)).map(topd)

    return rdd.partitionBy(4)


def mergeSets(set_list):
    result = []
    while len(set_list) > 0:
        cur_set = set_list.pop(0)
        intersect_idxs = [i for i in list(range(len(set_list) - 1, -1, -1)) if cur_set & set_list[i]]
        while intersect_idxs:
            for idx in intersect_idxs:
                cur_set = cur_set | set_list[idx]

            for idx in intersect_idxs:
                set_list.pop(idx)

            intersect_idxs = [i for i in list(range(len(set_list) - 1, -1, -1)) if cur_set & set_list[i]]

        result = result + [cur_set]
    return result


def Local_DBSCAN(df, eps, minpts):
    dfdata = df.copy()
    dfdata.columns = ["feature1_a", "feature2_a"]
    dfdata["id_a"] = range(1, len(df) + 1)
    dfdata = dfdata.set_index("id_a", drop=False)
    dfdata = dfdata.reindex(columns=["id_a", "feature1_a", "feature2_a"])
    dfdata.head()

    dfpairs = pd.DataFrame(columns=["id_a", "id_b", "distance_ab"])

    q = dfdata.loc[:, ["feature1_a", "feature2_a"]].values  # 坐标
    for i in dfdata.index:
        p = dfdata.loc[i, ["feature1_a", "feature2_a"]].values
        print(p)
        dfab = dfdata[["id_a"]].copy()
        dfab["id_b"] = i
        dfab["distance_ab"] = np.sqrt(np.sum((p - q) ** 2, axis=1))  # compute the distance

        dfpairs = pd.concat([dfpairs, dfab])

    dfnears = dfpairs.query(f"distance_ab<{eps}")  # 3 spiral

    dfglobs = dfnears.groupby("id_a").agg({"id_b": [len, set]})
    dfglobs.columns = ["neighbours_cnt", "neighbours"]
    dfglobs = dfglobs.query(f"neighbours_cnt>={minpts}")  # 2 spiral
    dfglobs = dfglobs.reset_index()

    # 找到核心点id
    core_ids = set(dfglobs["id_a"])
    dfcores = dfglobs.copy()

    # 剔除非核心点的id
    dfcores["neighbours"] = [x & core_ids for x in dfcores["neighbours"]]

    set_list = list(dfcores["neighbours"])
    result = mergeSets(set_list)
    core_clusters = {i: s for i, s in enumerate(result)}
    print(core_clusters)

    core_map = {}
    for k, v in core_clusters.items():
        core_map.update({vi: k for vi in v})

    cluster_map = {}
    for i in range(len(dfglobs)):
        id_a = dfglobs["id_a"][i]
        neighbours = dfglobs["neighbours"][i]
        cluster_map.update({idx: core_map[id_a] for idx in neighbours})
    dfdata["cluster_id"] = [cluster_map.get(id_a, -1) for id_a in dfdata["id_a"]]

    dfdata.plot.scatter('feature1_a', 'feature2_a', s=100,
                        c=list(dfdata['cluster_id']), cmap='rainbow', colorbar=False,
                        alpha=0.6, title='Hands DBSCAN Cluster Result ')
    return dfdata

def plotResult(data,x='feature1',y='feature2'):
    scatter = plt.scatter(list(data[x]), list(data[y]), c=list(data['cluster_id']))
    plt.legend(*scatter.legend_elements(),
                    loc="lower left", title="Classes")
    plt.show()





def clustering(x):
    l = []
    global eps
    global minpts

    for e in x:
        df = e
        l.append(df[1])
    dfdata = l[0]
    output = Local_DBSCAN(dfdata, eps.value, minpts.value)

    return [output]





if __name__=="__main__":
    spark = SparkSession\
        .builder\
        .appName("5003")
        .getOrCreate()
    origin_data = np.array(load_data_label('./spiral.txt'))
    train_data = origin_data[:, :2]

    preprocess_rdd = dataProcessing(train_data)
    eps = 3
    minpts = 2
    eps = sc.broadcast(eps)
    minpts = sc.broadcast(minpts)
    res = preprocess_rdd.mapPartitions(clustering)
    # res.count()
    res_0 = res.take(4)[0]
    res_1 = res.take(4)[1]
    res_1['cluster_id'] += (max(res_0['cluster_id']) + 1)
    res_2 = res.take(4)[2]
    res_2['cluster_id'] += (max(res_1['cluster_id']) + 1)
    res_3 = res.take(4)[3]
    res_3['cluster_id'] += (max(res_2['cluster_id']) + 1)
    res = pd.concat([res_0, res_1, res_2, res_3])
    res = res.rename(columns={'id_a': 'id', 'feature1_a': 'feature1', 'feature2_a': 'feature2'})
    res['coord'] = res.apply(lambda x: (x.feature1, x.feature2), axis=1)

    output = res.copy()
    coords = set(output['coord'])
    for coord in coords:
        clusters = set(output.loc[output.coord==coord, 'cluster_id'])
        min_cluster = min(clusters)
        output.loc[output.cluster_id.isin(clusters),'cluster_id'] = min_cluster
    output = output.drop(columns=['id'])
    output = output.drop_duplicates()
    plotResult(output)

    spark.stop()
