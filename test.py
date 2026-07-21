import tkinter
from tkinter import ttk

from mpmath.libmp import from_npfloat

root = tkinter.Tk()
root.title('test')
root.geometry('1400x900')

header = tkinter.Frame(root,bg='#2563eb',height=80)
header.pack(fill='x',padx=10,pady=5)
header.pack_propagate(False)

title_label = tkinter.Label(
    header,
    text='qqqq',
    # font=('newspaper',15,'bold'),
)
title_label.pack(side='left',padx=20,pady=10)

control_frame = tkinter.Frame(header,bg='#2585eb')
control_frame.pack(side='right',padx=20)

tkinter.Label(
    control_frame,
    text='www'
).pack(side='left',padx=5)

dataset_var = tkinter.StringVar(value='rubber')
dataset_combo = ttk.Combobox(
    control_frame,
    textvariable=dataset_var,
    values=['r','p']
)
dataset_combo.pack(side='left',padx=5)

notebook = ttk.Notebook(root)
notebook.pack(fill='both', expand=True, padx=10, pady=5)

upload_frame = tkinter.Frame(notebook,bg='white')
notebook.add(upload_frame,text='111')

center_frame = tkinter.Frame(upload_frame,bg='pink')
center_frame.place(relx=0.5,rely=0.5,anchor='center')

preview_frame = tkinter.Frame(center_frame, bg='yellow', relief='solid', borderwidth=2)
preview_frame.pack(pady=20)
image_preview_label = tkinter.Label(
            preview_frame,
            text='点击下方按钮上传图片\n支持 JPG, PNG 格式\n推荐尺寸 512×512',
            font=('newspaper', 12),
    bg='blue',
            width=80,
            height=30
        )
image_preview_label.pack(padx=30, pady=30)

upload_btn = tkinter.Button(
    center_frame,
    text='222',
    padx=30,
    pady=10,
    cursor='hand2'
)
upload_btn.pack(pady=10)

next_btn = tkinter.Button(
            center_frame,
            text='下一步: 选择模型 →',
            bg='#10b981',
            fg='white',
            font=('newspaper', 12, 'bold'),
    command=lambda :notebook.select(1),
            padx=20,
            pady=8
        )
next_btn.pack(pady=20)

#--------------------------

batch_frame = tkinter.Frame(notebook, bg='white')
notebook.add(batch_frame, text='222')

center_frame2 = tkinter.Frame(batch_frame, bg='green')
center_frame2.place(relx=0.5, rely=0.5, anchor='center')

title = tkinter.Label(
    center_frame2,
    text='2222',
    bg='pink'
)
title.pack(pady=20)

info = tkinter.Label(
            center_frame2,
            text='可以选择文件夹或批量选择图片文件',
            font=('newspaper', 12),
            bg='orange',
        )
info.pack(pady=10)

btn_frame = tkinter.Frame(center_frame2, bg='gray')
btn_frame.pack(pady=20)

folder_btn = tkinter.Button(
            btn_frame,
            text='222',
            padx=30,
            pady=15
        )
folder_btn.pack(side='left', padx=10)

files_btn = tkinter.Button(
            btn_frame,
            text='333',
            padx=30,
            pady=15
        )
files_btn.pack(side='left', padx=10)

next_btn = tkinter.Button(
            center_frame2,
            text='zuo',
            command=lambda: notebook.select(0),
            bg='#10b981',
            fg='white',
            font=('newspaper', 12, 'bold'),
            padx=20,
            pady=8
        )
next_btn.pack(pady=10)

#----------------------------
inference_frame = tkinter.Frame(notebook, bg='white')
notebook.add(inference_frame, text='333')

canvas = tkinter.Canvas(inference_frame, bg='pink')
scrollbar = ttk.Scrollbar(inference_frame, orient='vertical', command=canvas.yview)
scrollable_frame = tkinter.Frame(canvas, bg='yellow')

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)

canvas.create_window((0,0),window=scrollable_frame,anchor='nw')
canvas.configure(yscrollcommand=scrollbar.set)

canvas.pack(side='left', fill='both', expand=True)
scrollbar.pack(side='right', fill='y')

model_frame = tkinter.LabelFrame(
            scrollable_frame,
            text='选择推断模型',
            font=('newspaper', 14, 'bold'),
            bg='green',
            padx=20,
            pady=20
        )
model_frame.pack(fill='x', padx=20, pady=20)

#-------------------------
statusbar = tkinter.Label(
            root,
            text='就绪',
            relief='sunken',
            anchor='w',
            bg='#f3f4f6',
            font=('newspaper', 9)
        )
statusbar.pack(side='bottom', fill='x')


root.mainloop()