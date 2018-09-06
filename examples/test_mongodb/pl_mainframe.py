from __future__ import print_function, absolute_import
import tkinter as tk
from tkinter import ttk
import tkinter.font as tkFont
from tkinter.filedialog import *
import shutil

def parameterset():
    global test
    # global train_batch
    #   train_batch=100

    paraset = tk.Tk()
    paraset.wm_title("Parameters Setting")
    paraset.geometry("600x400+300+100")

    sysfont = tkFont.Font(size=18)
    menubar = tk.Menu(paraset)

    fmenu1 = tk.Menu(paraset)
    for item in ['New', 'Open', 'Save', 'Save as']:
        fmenu1.add_command(label=item)

    fmenu2 = tk.Menu(paraset)
    for item in ['Copy', 'Paste', 'Cut']:
        fmenu2.add_command(label=item)

    fmenu3 = tk.Menu(paraset)
    for item in ['Datebase', 'Output']:
        fmenu3.add_command(label=item)

    fmenu4 = tk.Menu(paraset)
    for item in ["Version", "About"]:
        fmenu4.add_command(label=item)

    menubar.add_cascade(label="file", menu=fmenu1)
    menubar.add_cascade(label="edit", menu=fmenu2)
    menubar.add_cascade(label="database", menu=fmenu3)
    menubar.add_cascade(label="about", menu=fmenu4)

    paraset['menu'] = menubar

    def para_confirm():
        global train_batch
        global learningrate
        global max_epoch

        train_batch = int(trainbatch.get())
        learningrate = float(lr.get())
        max_epoch = int(ep.get())
        test = trainbatch.get()
        print("train_batchsize: %s\nlearning rate: %s\nepoch: %s\n" % (trainbatch.get(), lr.get(), ep.get()))



    ttk.Label(paraset, text="GPU_device_number").grid(column=0, row=0)
    ttk.Label(paraset, text="batchsize").grid(column=0, row=1)
    ttk.Label(paraset, text="learning rate").grid(column=0, row=2)
    ttk.Label(paraset, text="epoch").grid(column=0, row=3)
    ttk.Label(paraset, text="loss function").grid(column=0, row=4)

    trainbatch = ttk.Entry(paraset)
    lr = ttk.Entry(paraset)
    ep = ttk.Entry(paraset)

    trainbatch.grid(row=1, column=1)
    lr.grid(row=2, column=1)
    ep.grid(row=3, column=1)

    ttk.Button(paraset, text='Quit', command=maintable).grid(row=5, column=0, pady=4)  #paraset.quit
    ttk.Button(paraset, text='Confirm', command=para_confirm).grid(row=5, column=1, pady=4)

    number = tk.StringVar()
    numberChosen = ttk.Combobox(paraset, width=18, textvariable=number)
    numberChosen['values'] = ('cross_entropy_loss', 'cross_entropy_loss and hard_triplet_loss', 'hard_triplet_loss')
    numberChosen.grid(column=1, row=4)
    numberChosen.current(1)

    GPU_device = tk.StringVar()
    GPU_deviceChosen = ttk.Combobox(paraset, width=18, textvariable=GPU_device)
    GPU_deviceChosen['values'] = ('0', '1', '2', '3')  # 设置下拉列表的值
    GPU_deviceChosen.grid(column=1, row=0)  # 设置其在界面中出现的位置  column代表列   row 代表行
    GPU_deviceChosen.current(0)  # 设置下拉列表默认显示的值，0为 numberChosen['values'] 的下标值

    paraset.mainloop()









def trainning():
    global test
    top = tk.Tk()
    top.title("Python GUI")  # 添加标题
    top.geometry("600x400+300+100")

    sysfont = tkFont.Font(size=18)  # 创建一个菜单项，类似于导航栏
    menubar = tk.Menu(top)

    fmenu1 = tk.Menu(top)
    for item in ['New', 'Open', 'Save', 'Save as']:
        fmenu1.add_command(label=item)

    fmenu2 = tk.Menu(top)
    for item in ['Copy', 'Paste', 'Cut']:
        fmenu2.add_command(label=item)

    fmenu3 = tk.Menu(top)
    for item in ['Datebase', 'Output']:
        fmenu3.add_command(label=item)

    fmenu4 = tk.Menu(top)
    for item in ["Version", "About"]:
        fmenu4.add_command(label=item)

    menubar.add_cascade(label="file", menu=fmenu1)
    menubar.add_cascade(label="edit", menu=fmenu2)
    menubar.add_cascade(label="database", menu=fmenu3)
    menubar.add_cascade(label="about", menu=fmenu4)

    top['menu'] = menubar

    ttk.Label(top, text="Platform").grid(column=0, row=0, pady=2)
    ttk.Label(top, text="Loss").grid(column=0, row=1, pady=2)
    ttk.Label(top, text="Database").grid(column=0, row=2, pady=2)
    ttk.Label(top, text="model").grid(column=0, row=3, pady=2)
    ttk.Label(top, text="GPU_device_number").grid(column=0, row=4, pady=2)
    ttk.Label(top, text="batchsize").grid(column=0, row=5, pady=2)
    ttk.Label(top, text="learning rate").grid(column=0, row=6, pady=2)
    ttk.Label(top, text="epoch").grid(column=0, row=7, pady=2)


    trainbatch = ttk.Entry(top)
    lr = ttk.Entry(top)
    ep = ttk.Entry(top)

    trainbatch.grid(row=5, column=1, pady=1)
    lr.grid(row=6, column=1, pady=1)
    ep.grid(row=7, column=1, pady=1)


# ttk.Label(top, text="Chooes an algprithm").grid(column=0, row=0)  # 添加一个标签，并将其列设置为1，行设置为0
# ttk.Label(top, text="Platform:").grid(column=1, row=0)  # 设置其在界面中出现的位置  column代表列   row 代表行

    def entrygo(*args):  # 处理事件，*args表示可变参数
        global arch
        arch = nameEntered.get()  # 打印选中的值


    def upload_dataset():
        oct(os.stat('/media/saber/').st_mode)[-3:]
        filename = askopenfilename(title="Choose Uploading Dataset", initialdir='/home/saber')
        old = filename
        new = "/media/saber/DATASET/dataset"
        shutil.copy(old, new)
        filepath = new + '/' +filename.split('/')[-1]
        print(filename)
        print(filepath)
        k = os.path.exists(filepath)
        if k:
            print("Uploading Success")
        else:
            print("Uploading Failed")


# 按钮
    action = ttk.Button(top, text="Uploading Dataset", command=upload_dataset)  # 创建一个按钮, text：显示按钮上面显示的文字, command：当这个按钮被点击之后会调用command函数
    action.grid(column=2, row=1)  # 设置其在界面中出现的位置  column代表列   row 代表行

    trainning = ttk.Button(top, text="entry", command=entrygo)  # 创建一个按钮, text：显示按钮上面显示的文字, command：当这个按钮被点击之后会调用command函数
    trainning.grid(column=2, row=2)  # 设置其在界面中出现的位置  column代表列   row 代表行


# 文本框
# name = tk.StringVar()  # StringVar是Tk库内部定义的字符串变量类型，在这里用于管理部件上面的字符；不过一般用在按钮button上。改变StringVar，按钮上的文字也随之改变。
# nameEntered = ttk.Entry(top, width=12, textvariable=name)  # 创建一个文本框，定义长度为12个字符长度，并且将文本框中的内容绑定到上一句定义的name变量上，方便clickMe调用
# nameEntered.grid(column=0, row=1)  # 设置其在界面中出现的位置  column代表列   row 代表行
# nameEntered.focus()  # 当程序运行时,光标默认会出现在该文本框中
# nameEntered.bind("<<ComboboxSelected>>", entrygo)  # 绑定事件,(下拉列表框被选中时，绑定go()函数)


    def modelchoose(*args):  # 处理事件，*args表示可变参数
        global arch
        arch = ModelChosen.get()  # 打印选中的值


    def datachoose(*args):
        global database
        database = DatabaseChosen.get()


    def losschoose(*args):
        global loss
        loss = LossChosen.get()


    def testvalue():
        print(test)

    def lableon():

        global feature_num
        global matrix_num
        global batch_time
        global batch_size
        global rank1_result
        global rank5_result
        global rank10_result
        global rank20_result
        global cmc_result

        l1 = ttk.Label(top, text='Extracted features for gallery set, obtained' + str(feature_num) + 'by' + str(
            matrix_num)).grid(row=6, column=3)
        l2 = ttk.Label(top, text='BatchTime(s)' + str(batch_time)).grid(row=7, column=3)
        l3 = ttk.Label(top, text='BatchSize(img)' + str(batch_size)).grid(row=8, column=3)
        l4 = ttk.Label(top, text='mAP:' + str(cmc_result)).grid(row=9, column=3)
        r1 = ttk.Label(top, text='Rank1:' + str(rank1_result)).grid(row=10, column=3)
        r2 = ttk.Label(top, text='Rank5:' + str(rank5_result)).grid(row=11, column=3)
        r3 = ttk.Label(top, text='Rank10:' + str(rank10_result)).grid(row=12, column=3)
        r4 = ttk.Label(top, text='Rank20:' + str(rank20_result)).grid(row=13, column=3)






# button被点击之后会被执行
# def clickMe():  # 当acction被点击时,该函数则生效
#    action.configure(text='Hello ' + name.get())  # 设置button显示的内容
#   action.configure(state='disabled')  # 将按钮设置为灰色状态，不可使用状态


    ttk.Button(top, text='Parameters Setting', command=parameterset).grid(row=9, column=0, pady=4)
    ttk.Button(top, text='Training', command=reid).grid(row=9, column=1, pady=4)
    ttk.Button(top, text='lableon', command=lableon).grid(row=9, column=2, pady=4)
    ttk.Button(top, text='load pretrained model', command=testvalue).grid(row=6, column=2, pady=8)

    number = tk.StringVar()
    numberChosen = ttk.Combobox(top, width=18, textvariable=number)
    numberChosen['values'] = ('Tensorflow', 'Pytorch', 'mudeep', 'resnet50', 'resnet101')  # 设置下拉列表的值
    numberChosen.grid(column=1, row=1, pady=5)  # 设置其在界面中出现的位置  column代表列   row 代表行
    numberChosen.current(0)  # 设置下拉列表默认显示的值，0为 numberChosen['values'] 的下标值
    numberChosen.bind("<<ComboboxSelected>>", entrygo)  # 绑定事件,(下拉列表框被选中时，绑定go()函数)

    Platform = tk.StringVar()
    PlatformChosen = ttk.Combobox(top, width=18, textvariable=Platform)
    PlatformChosen['values'] = ('Tensorflow', 'Pytorch', 'Mxnet')  # 设置下拉列表的值
    PlatformChosen.grid(column=1, row=0, pady=5)  # 设置其在界面中出现的位置  column代表列   row 代表行
    PlatformChosen.current(1)  # 设置下拉列表默认显示的值，0为 numberChosen['values'] 的下标值

    Loss = tk.StringVar()
    LossChosen = ttk.Combobox(top, width=18, textvariable=Loss)
    LossChosen['values'] = ('softmax', 'triplet', 'oim')  # 设置下拉列表的值
    LossChosen.grid(column=1, row=1, pady=5)  # 设置其在界面中出现的位置  column代表列   row 代表行
    LossChosen.current(0)  # 设置下拉列表默认显示的值，0为 numberChosen['values'] 的下标值
    LossChosen.bind("<<ComboboxSelected>>", losschoose)

    Database = tk.StringVar()
    DatabaseChosen = ttk.Combobox(top, width=18, textvariable=Database)
    DatabaseChosen['values'] = ('market1501', 'cuhk01', 'cuhk03', 'dukemtmc', 'msmt17', 'VIPeR', 'GRID')  # 设置下拉列表的值
    DatabaseChosen.grid(column=1, row=2, pady=5)  # 设置其在界面中出现的位置  column代表列   row 代表行
    DatabaseChosen.current(0)  # 设置下拉列表默认显示的值，0为 numberChosen['values'] 的下标值
    DatabaseChosen.bind("<<ComboboxSelected>>", datachoose)

    Model = tk.StringVar()
    ModelChosen = ttk.Combobox(top, width=18, textvariable=Model)
    ModelChosen['values'] = (
    'resnet50', 'resnet101', 'resnet50M', 'densenet121', 'squeezenet', 'mobilenetv2', 'shufflenet', 'xception',
    'inceptionv4', 'nasnsetmobile', 'inceptionresnetv2', 'mudeep', 'hacnn')  # 设置下拉列表的值
    ModelChosen.grid(column=1, row=3, pady=5)  # 设置其在界面中出现的位置  column代表列   row 代表行
    ModelChosen.current(0)  # 设置下拉列表默认显示的值，0为 numberChosen['values'] 的下标值
    ModelChosen.bind("<<ComboboxSelected>>", modelchoose)

    GPU_device = tk.StringVar()
    GPU_deviceChosen = ttk.Combobox(top, width=18, textvariable=GPU_device)
    GPU_deviceChosen['values'] = ('0', '1', '2', '3')  # 设置下拉列表的值
    GPU_deviceChosen.grid(column=1, row=4, pady=5)  # 设置其在界面中出现的位置  column代表列   row 代表行
    GPU_deviceChosen.current(0)  # 设置下拉列表默认显示的值，0为 numberChosen['values'] 的下标值


    top.mainloop()



def loadcheckpoint():
    load = tk.Tk()
    load.title("Python GUI")  # 添加标题
    load.geometry("600x400+300+100")

    sysfont = tkFont.Font(size=18)
    menubar = tk.Menu(load)

    fmenu1 = tk.Menu(load)
    for item in ['New', 'Open', 'Save', 'Save as']:
        fmenu1.add_command(label=item)

    fmenu2 = tk.Menu(load)
    for item in ['Copy', 'Paste', 'Cut']:
        fmenu2.add_command(label=item)

    fmenu3 = tk.Menu(load)
    for item in ['Datebase', 'Output']:
        fmenu3.add_command(label=item)

    fmenu4 = tk.Menu(load)
    for item in ["Version", "About"]:
        fmenu4.add_command(label=item)

    menubar.add_cascade(label="file", menu=fmenu1)
    menubar.add_cascade(label="edit", menu=fmenu2)
    menubar.add_cascade(label="database", menu=fmenu3)
    menubar.add_cascade(label="about", menu=fmenu4)

    load['menu'] = menubar

    def checkpoint():
        global log_dir
        global resume
        global resumepath

        oct(os.stat('/media/saber/').st_mode)[-3:]
        log_dir = askopenfilename(title="Choose Uploading Dataset", initialdir='/media/saber/DATASET/reid-demo/log')
        if log_dir:
            resume = True
            resumepath = log_dir


    def bestmodel():
        global log_dir
        global resume
        global resumepath

        oct(os.stat('/media/saber/').st_mode)[-3:]
        log_dir = askopenfilename(title="Choose Uploading Dataset", initialdir='/media/saber/DATASET/reid-demo/log')
        if log_dir:
            resume = True
            resumepath = log_dir


    ttk.Button(load, text='Loading Best Models', command=bestmodel).grid(row=0, column=0, pady=4)
    ttk.Button(load, text='Loading CheckPoint', command=checkpoint).grid(row=0, column=1, pady=4)
    #    ttk.Button(main, text='Searching Logs', command=lableon).grid(row=0, column=2, pady=4)

    load.mainloop()




def maintabel():

    main = tk.Tk()
    main.title("Python GUI")  # 添加标题
    main.geometry("600x400+300+100")

    sysfont = tkFont.Font(size=18)
    menubar = tk.Menu(main)

    fmenu1 = tk.Menu(main)
    for item in ['New', 'Open', 'Save', 'Save as']:
        fmenu1.add_command(label=item)

    fmenu2 = tk.Menu(main)
    for item in ['Copy', 'Paste', 'Cut']:
        fmenu2.add_command(label=item)

    fmenu3 = tk.Menu(main)
    for item in ['Datebase', 'Output']:
        fmenu3.add_command(label=item)

    fmenu4 = tk.Menu(main)
    for item in ["Version", "About"]:
        fmenu4.add_command(label=item)

    menubar.add_cascade(label="file", menu=fmenu1)
    menubar.add_cascade(label="edit", menu=fmenu2)
    menubar.add_cascade(label="database", menu=fmenu3)
    menubar.add_cascade(label="about", menu=fmenu4)

    main['menu'] = menubar



    ttk.Button(main, text='Choose CheckPoint Model', command=trainning).grid(row=0, column=0, pady=4)
    ttk.Button(main, text='Loading Exist Model', command=loadcheckpoint).grid(row=0, column=1, pady=4)

    main.mainloop()


maintabel()