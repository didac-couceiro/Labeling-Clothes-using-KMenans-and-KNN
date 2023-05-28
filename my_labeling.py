__authors__ = 'TO_BE_FILLED'
__group__ = 'TO_BE_FILLED'

import numpy as np
from Kmeans import*
import matplotlib.pyplot as plt
from utils_data import read_dataset, read_extended_dataset, crop_images, visualize_retrieval, Plot3DCloud
import time as time
from KNN import*
from scipy.stats import mode

if __name__ == '__main__':

    # Load all the images and GT

    train_imgs_grey, train_class_labels, train_color_labels, test_imgs_grey, test_class_labels, \
    test_color_labels = read_dataset(root_folder='./images/', gt_json='./images/gt.json', with_color=False)

    train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, \
        test_color_labels = read_dataset(root_folder='./images/', gt_json='./images/gt.json')


    # List with all the existent classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))

    # Load extended ground truth
    imgs, class_labels, color_labels, upper, lower, background = read_extended_dataset()
    cropped_images = crop_images(imgs, upper, lower)
    #You can start coding your functions here


def Kmean_statistics(kmeans, Kmax):
    it = []
    w = []
    ks = []
    for i in range(2, Kmax + 1):
        guard = np.empty(5, dtype="U30")
        kmeans.K = i
        inicio = time.time()
        kmeans.fit()
        final = time.time() - inicio
        kmeans.withinClassDistance()
        a = kmeans.WCD
        ks.append(i)
        w.append(a)
        guard[0] = i
        guard[1] = final
        guard[2] = kmeans.num_iter
        guard[3] = a
        it.append(guard[2])
        print('Amb k =', guard[0], 'ha tardar en convergir:', guard[1], 'con WCD =', guard[3], 'en la iteracion:',
              guard[2])

    plt.xlabel('Eje K')
    plt.ylabel('Eje WCD')
    plt.scatter(ks, it, color='blue')
    plt.show()


def Get_shape_accuracy(knn,ground):
    #el knn x ejemplo tiene 30imag te devuelve una prediccion con 30 etiquetas
    #el ground es una lista de=dimension que knn pero los valores de ese vector son etiq.reales

    longitud=len(knn)
    contador=0

    for i in range(longitud):
        if(knn[i]==ground[i]):# comprovem element per element
            contador+=1
    return 100 * (contador/longitud)

def Get_color_accuracy(kmeans_labels, ground_truth):
    porc = 0
    for pred, real in zip(kmeans_labels, ground_truth):
        #mirem la dimensio de la interseccio entre les nostres etiquetes i les reals
        inter=len(np.intersect1d(pred, real))
        maxim =(max(len(np.unique(pred)), len(real)))
        porc+= inter / maxim
    return (porc / len(kmeans_labels)) * 100


"""
prova pel get color accuracy
colors = []
for im in imgs:
    km = KMeans(im, 5)
    km.fit()
    colors.append(get_colors(km.centroids))
print('El percentatge obtingut és: '+ str(Get_color_accuracy(colors,color_labels))+'%')
"""
"""
#prova pel get shape accuracy

knn = KNN(train_imgs_grey,train_class_labels)
shapes = knn.predict(test_imgs_grey,4)
print(shapes[10:20])
print(test_class_labels[10:20])
print(str(Get_shape_accuracy(shapes,test_class_labels))+'%')
"""
"""
#prova pel kmean_statistics
for i, im in enumerate(imgs[:5]):
    km = KMeans(im,4)
    Kmean_statistics(km,10)
"""

def retrieval_by_color(list_images, predicted_colors, search, n):
        """
        Args:
            list_images: dataset of the images, obtained by the ground truth
            predicted_colors: list of the colors we have obtained after aply the K-means
            search: colors we want to search in the images
            n: number of images we want to see

        Return:
            Return a list of the index that have these colors
        """
        indicesToPrint = []

        if type(search) != list: search = [search]

        for index, element in enumerate(predicted_colors):
            auxList = np.isin(search, element)
            if np.all(auxList): # check that all the colors we want are in the sample
                indicesToPrint.append(index)

        random.shuffle(indicesToPrint)
        imagesGiven = list_images[indicesToPrint]
        visualize_retrieval(imagesGiven, n)
        return indicesToPrint

def retrieval_by_shape(list_images, predicted_shapes, search, n):
    """
    Args:
        list_images: dataset of the images, obtained by the ground truth
        predicted_shapes: list of the shapes we have obtained after aply the KNN
        search: shape we want to search in the images
        n: number of images we want to see

    Return:
        Return a list of the indexs that have these shapes
    """

    indicesToPrint = []

    for index, element in enumerate(predicted_shapes):
        if search in element: # check that all the colors we want are in the sample
            print(index)
            indicesToPrint.append(index)

    random.shuffle(indicesToPrint)
    imagesGiven = list_images[indicesToPrint]
    visualize_retrieval(imagesGiven, n)
    return imagesGiven

def retrieval_combined(list_images, predicted_colors, predicted_shapes, search_color, search_shape, n):
    """
    Args:
        list_images: dataset of the images, obtained by the ground truth
        predicted_colors: list of the colors we have obtained after aply the Kmeans
        predicted_shapes: list of the shapes we have obtained after aply the KNN
        search_color: colors we want to search in the images
        search_shapes: shape we want to search in the images
        n: number of images we want to see

    Return:
        Return a list of the indexs that have these shapes
    """
    indicesColor = []
    indicesToPrint = []

    for index, color in enumerate(predicted_colors):
        auxList = np.isin(search_color, color)
        if np.all(auxList):  # check that all the colors we want are in the sample
            indicesColor.append(index)

    list_images = np.array(list_images)[indicesColor]
    predicted_shapes = np.array(predicted_shapes)[indicesColor]

    for index, shape in enumerate(predicted_shapes):
        if shape in search_shape:
            indicesToPrint.append(index)

    random.shuffle(indicesToPrint)
    imagesGiven = list_images[indicesToPrint]
    visualize_retrieval(imagesGiven, n)

    return indicesToPrint

def retrieval_by_colors_combination(list_images, predicted_colors, search, n, search_type=1):
    """
    Args:
        list_images: dataset of the images, obtained by the ground truth
        predicted_colors: list of the colors we have obtained after aply the K-means
        search: combination of colors we want to search in the images
        search_type: you can choose if you want samples where there is only all this colors
                    or samples where there are combinations of this colors but there is no need
                    of all the colors in the same sample.
                    If search_type = 0, it shows only samples that have all the colors.
                    If search_type = 1, it shows all the samples that have at least one color ordered
                    by more colors to less
                    If search_type = 2, it shows all the samples that have at least one color
                    ordered randomly
        n: number of images we want to see


    Return:
        Return a list of the index that have these colors
    """
    indicesToPrint = []
    indicesAux = {}


    if type(search) != list: search = [search]

    for index, element in enumerate(predicted_colors):
        auxList = np.isin(search, element)
        #if(np.sum(auxList)>1):
            #print(np.sum(auxList))
        if np.all(auxList): # check that all the colors we want are in the sample
            indicesToPrint.append(index)

        elif np.sum(auxList) >= 1 and search_type != 0:
            if(search_type == 1):
                key = np.sum(auxList)
                if key in indicesAux:
                    indicesAux[key].append(index)
                else:
                    indicesAux[key] = [index]

            else:
                indicesToPrint.append(index)

    if search_type == 1:
        # Sort the keys in descending order and reverse the order
        sorted_keys = sorted(indicesAux.keys(), reverse=False)
        for key in reversed(sorted_keys):
            list_to_merge = indicesAux[key]
            for i in list_to_merge:
                indicesToPrint.append(i)

        imagesGiven = list_images[indicesToPrint]
        visualize_retrieval(imagesGiven, n)
    else:
        random.shuffle(indicesToPrint)
        imagesGiven = list_images[indicesToPrint]
        visualize_retrieval(imagesGiven, n)

    return indicesToPrint


# load our knn
knn = KNN(train_imgs, train_class_labels)
label_results = knn.predict(test_imgs, 6)

# load our kmeans
colors = []
for im in test_imgs:
    km = KMeans(im, 4)
    km.fit()
    colors.append(get_colors(km.centroids))
    # print(get_colors(km.centroids))

# retrieval_by_color(test_imgs, colors, ["Blue"], 10)
# retrieval_by_color(test_imgs, colors, ["Yellow", "Blue"], 10)
# retrieval_by_shape(test_imgs, label_results, "Handbags", 10)
# retrieval_by_shape(test_imgs, label_results, "Dresses", 10)
# retrieval_combined(test_imgs, colors, label_results, ["Red"], "Handbags",5)
# retrieval_combined(test_imgs, colors, label_results, ["Blue", "Yellow"], "Flip Flops",5)
retrieval_by_colors_combination(test_imgs, test_color_labels, ["Blue", "Yellow", "Pink", "Brown", "Orange"], 10, 1)


temps = []
acc = []

def comparison_knn_distances():
    print(f"\nDistància usada: Euclidiana")
    t0 = time.perf_counter()
    knn = KNN(train_imgs_grey,train_class_labels)
    shapes = knn.predict(test_imgs_grey,4)
    print(shapes[10:20])
    print(test_class_labels[10:20])
    accu = Get_shape_accuracy(shapes,test_class_labels)
    print(str(accu)+'%')
    t = time.perf_counter() - t0
    print(f'Temps:{t}')
    temps.append(t)
    acc.append(accu)

    print(f"\nDistància usada: Cosine Similarity")
    t0 = time.perf_counter()
    knn = KNN(train_imgs_grey,train_class_labels,distance='Cosine Similarity')
    shapes = knn.predict(test_imgs_grey,4)
    print(shapes[10:20])
    print(test_class_labels[10:20])
    accu = Get_shape_accuracy(shapes,test_class_labels)
    print(str(accu)+'%')
    t = time.perf_counter() - t0
    print(f'Temps:{t}')
    temps.append(t)
    acc.append(accu)

    print(f"\nDistància usada: Manhattan")
    t0 = time.perf_counter()
    knn = KNN(train_imgs_grey,train_class_labels,distance='Manhattan')
    shapes = knn.predict(test_imgs_grey,4)
    print(shapes[10:20])
    print(test_class_labels[10:20])
    accu = Get_shape_accuracy(shapes,test_class_labels)
    print(str(accu)+'%')
    t = time.perf_counter() - t0
    print(f'Temps:{t}')
    temps.append(t)
    acc.append(accu)

    print(f"\nDistància usada: Jaccard")
    t0 = time.perf_counter()
    knn = KNN(train_imgs_grey,train_class_labels,distance='Jaccard')
    shapes = knn.predict(test_imgs_grey,4)
    print(shapes[10:20])
    print(test_class_labels[10:20])
    accu = Get_shape_accuracy(shapes,test_class_labels)
    print(str(accu)+'%')
    t = time.perf_counter() - t0
    print(f'Temps:{t}')
    temps.append(t)
    acc.append(accu)

    print(f"\nDistància usada: Chebyshev")
    t0 = time.perf_counter()
    knn = KNN(train_imgs_grey,train_class_labels,distance='Chebyshev')
    shapes = knn.predict(test_imgs_grey,4)
    print(shapes[10:20])
    print(test_class_labels[10:20])
    accu = Get_shape_accuracy(shapes,test_class_labels)
    print(str(accu)+'%')
    t = time.perf_counter() - t0
    print(f'Temps:{t}')
    temps.append(t)
    acc.append(accu)
    print("\n\n")
    print(temps)
    print(acc)

"""
#prova pel kmean_statistics
for i, im in enumerate(imgs[:5]):
    km = KMeans(im,4)
    Kmean_statistics(km,10)
"""
def plot_comparison_knn():
    temps1 = [10.420848400040995,8.873893700016197,7.280299700039905,57.47214430000167,26.764058899949305]
    acc1 = [90.7168037602820,87.66157461809637,90.83431257344301,89.77673325499413,25.499412455934195]

    temps2 = [9.22151100001065, 14.712873200012837, 14.99872689996846, 87.16076329996577, 41.84278920001816]
    acc2 = [90.71680376028202, 87.66157461809637, 90.83431257344301, 89.77673325499413, 25.499412455934195]

    temps3 = [10.479671600041911, 14.348869499983266, 14.569774799980223, 99.46210810000775, 41.704148399992846]
    acc3 = [90.71680376028202, 87.66157461809637, 90.83431257344301, 89.77673325499413, 25.499412455934195]

    methods = ['Euclidean','Cosine','Manhattan','Jaccard','Chebyshev']
    a = np.mean([acc1,acc2,acc3],axis=0)
    t = np.mean([temps1,temps2,temps3],axis=0)

    print(a)


    plt.scatter(t, a)
    plt.xlabel('temps (s)')
    plt.ylabel('Accuracy (%)')
    plt.title('Comparació de les diferents distàncies')

    # Add labels to data points
    for i in range(len(methods)):
        plt.scatter(t[i], a[i], label=methods[i])

    # Display the plot
    plt.grid()
    plt.legend()
    plt.show()


def accuracy_col(K, init_cent):
    colors = []
    for im in imgs:
        km = KMeans(im, K,options = {'km_init':init_cent})
        km.fit()
        colors.append(get_colors(km.centroids))
    acc = Get_color_accuracy(colors,color_labels)
    return acc

def get_times_vs_col_accuracy(K):
    dict_acc = {'first':[],'random':[],'custom':[]}
    dict_time = {'first':[],'random':[],'custom':[]}

    init = ['first','random','custom']

    for k in range(2,K+1):
        for i in init:
            t0 = time.perf_counter()
            acu = accuracy_col(k,i)
            tf = time.perf_counter() - t0
            dict_acc[i].append(acu)
            dict_time[i].append(tf)

    return dict_acc,dict_time


def get_median_K(maxK,method):
    k = []
    for im in imgs[:30]:
        km = KMeans(im,options = {'km_init':'random','fitting':method})
        km.find_bestK(maxK)
        k.append(km.K)
    result =  mode(np.array(k))
    return result.mode[0]

def barchart_of_best_K():
    K_WCD = 5
    K_sil = 4
    K_FD = 10
    K_IC = 10

    t = []
    a = []

    t0 = time.perf_counter()
    colors = []
    for im in imgs:
        km = KMeans(im, K_WCD,options = {'km_init':'random'})
        km.fit()
        colors.append(get_colors(km.centroids))
    acc = Get_color_accuracy(colors,color_labels)
    tf = time.perf_counter() - t0
    t.append(tf)
    a.append(acc)

    t0 = time.perf_counter()
    colors = []
    for im in imgs:
        km = KMeans(im, K_sil,options = {'km_init':'random'})
        km.fit()
        colors.append(get_colors(km.centroids))
    acc = Get_color_accuracy(colors,color_labels)
    tf = time.perf_counter() - t0
    t.append(tf)
    a.append(acc)

    t0 = time.perf_counter()
    colors = []
    for im in imgs:
        km = KMeans(im, K_FD,options = {'km_init':'random'})
        km.fit()
        colors.append(get_colors(km.centroids))
    acc = Get_color_accuracy(colors,color_labels)
    tf = time.perf_counter() - t0
    t.append(tf)
    a.append(acc)

    t0 = time.perf_counter()
    colors = []
    for im in imgs:
        km = KMeans(im, K_IC,options = {'km_init':'random'})
        km.fit()
        colors.append(get_colors(km.centroids))
    acc = Get_color_accuracy(colors,color_labels)
    tf = time.perf_counter() - t0
    t.append(tf)
    a.append(acc)

    a = a[:-1]
    t = t[:-1]
    methods = [5, 4, 10]

    bar_width = 0.35

    bar_positions = np.arange(len(methods))

    fig, ax = plt.subplots()

    ax.bar(bar_positions - bar_width/2, t, bar_width, label='Temps')
    ax.bar(bar_positions + bar_width/2, a, bar_width, label='Accuracy')

    ax.set_xlabel('Valors de K')
    ax.set_ylabel('Valors de temps i accuracy')
    ax.set_title('Comparison of Time and Accuracy for Diferent Best K Methods')
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(methods)
    ax.legend()

    plt.show()

def centroid_comparison():
    k1 = KMeans(train_imgs[0], 4, options={'km_init':'first'})
    k1._init_centroids()
    k2 = KMeans(train_imgs[0], 4, options={'km_init':'random'})
    k2._init_centroids()
    k3 = KMeans(train_imgs[0], 4, options={'km_init':'kmeans++'})
    k3._init_centroids()

    visualize_init_centroid(k1)
    plt.show()
    visualize_init_centroid(k2)
    plt.show()
    visualize_init_centroid(k3)
    plt.show()
    k1.fit()
    Plot3DCloud(k1)
    plt.show()