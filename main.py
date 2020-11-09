import tkinter
from PIL import Image, ImageTk
from tkinter.filedialog import askopenfilename
import numpy as np
import cv2
import time


input_img_array = None
query_img_array = None
match_array = None
match_coor = None
diff_original = None
diff_save_path = "new_light_condition_1014/diff_results/"


def gaussian_low_pass_filter(image, d):
    coor_matrix = np.zeros((image.shape[0], image.shape[1], 2))
    center_point = tuple(map(lambda x: (x - 1) / 2, image.shape[0:2]))
    center = np.array(center_point)
    center = np.expand_dims(center, 0)
    center = np.expand_dims(center, 0)
    x_coors = np.expand_dims(np.arange(image.shape[0]), 1)
    y_coors = np.expand_dims(np.arange(image.shape[1]), 0)
    coor_matrix[:, :, 0] += x_coors
    coor_matrix[:, :, 1] += y_coors
    
    coor_matrix = coor_matrix.astype(np.float32)

    distance_matrix = np.sqrt(np.sum((coor_matrix - center) ** 2, axis=2))
    #     print(distance_matrix.shape)
    map_distance = np.exp(-(distance_matrix ** 2) / (2 * (d ** 2)))

    blur_img = np.zeros(image.shape)
    for i in range(3):
        f = np.fft.fft2(image[:, :, i])
        f_shift = np.fft.fftshift(f)
        blur_img[:, :, i] = np.abs(np.fft.ifft2(np.fft.ifftshift(f_shift * map_distance)))

    return blur_img


def draw_and_show_rectangle(input_img, query_img, coor):
    h, w = query_img.shape[0], query_img.shape[1]
    match_image = input_img.copy()
    match_image = cv2.rectangle(match_image, (coor[1], coor[0]), (coor[1] + w, coor[0] + h),
                                (0, 255, 150), 5)
    w_box = 400
    h_box = 400
    h, w = input_img.shape[0], input_img.shape[1]
    image_show = Image.fromarray(match_image)
    f1 = 1.0 * w_box / w
    f2 = 1.0 * h_box / h
    factor = min([f1, f2])
    resize_w = int(factor * w)
    resize_h = int(factor * h)
    image_show = image_show.resize((resize_w, resize_h), Image.ANTIALIAS)

    cover = ImageTk.PhotoImage(image=image_show)
    label_show_input_image.configure(image=cover)
    label_show_input_image.image = cover

    return 0


def show_diff_result(query_img, match_img):
    diff_img_array = (query_img - match_img + 127).astype(np.uint8)
    w_box = 200
    h_box = 200
    diff_img = Image.fromarray(diff_img_array)
    w, h = diff_img.size
    f1 = 1.0 * w_box / w
    f2 = 1.0 * h_box / h
    factor = min([f1, f2])
    resize_w = int(factor * w)
    resize_h = int(factor * h)
    img_resize = diff_img.resize((resize_w, resize_h), Image.ANTIALIAS)

    cover = ImageTk.PhotoImage(image=img_resize)
    label_show_diff_image.configure(image=cover)
    label_show_diff_image.image = cover

    return 0


def save_diff():
    save_name = save_file_name_entry.get()
    if save_name == "":
        save_name = time.strftime("%Y-%m-%d-%H%M%S", time.localtime(time.time()))
    if diff_original is not None:
        diff_save = cv2.cvtColor(diff_original, cv2.COLOR_RGB2BGR)
        save_path = diff_save_path + save_name + ".jpg"
        cv2.imwrite(save_path, diff_save)
    else:
        print("There is no diff image available.")
    
    return 0


def show_blurred_diff_result():
    blur_d = 150
    h, w = query_img_array.shape[0], query_img_array.shape[1]
    match_area = input_img_array[match_coor[0]:match_coor[0] + h, match_coor[1]:match_coor[1] + w].copy()
    blurred_match_img = gaussian_low_pass_filter(match_area, blur_d)
    blurred_query_img = gaussian_low_pass_filter(query_img_array, blur_d)

    diff_img_array = (blurred_query_img - blurred_match_img + 127).astype(np.uint8)
    w_box = 200
    h_box = 200
    diff_img = Image.fromarray(diff_img_array)
    w, h = diff_img.size
    f1 = 1.0 * w_box / w
    f2 = 1.0 * h_box / h
    factor = min([f1, f2])
    resize_w = int(factor * w)
    resize_h = int(factor * h)
    img_resize = diff_img.resize((resize_w, resize_h), Image.ANTIALIAS)

    cover = ImageTk.PhotoImage(image=img_resize)
    label_show_blurred_diff_image.configure(image=cover)
    label_show_blurred_diff_image.image = cover


def show_input_img():
    global input_img_array
    w_box = 400
    h_box = 400
    img_path = askopenfilename()
    input_img = Image.open(img_path)
    input_img_array = np.asarray(input_img)

    w, h = input_img.size
    f1 = 1.0 * w_box / w
    f2 = 1.0 * h_box / h
    factor = min([f1, f2])
    resize_w = int(factor * w)
    resize_h = int(factor * h)
    img_resize = input_img.resize((resize_w, resize_h), Image.ANTIALIAS)

    cover = ImageTk.PhotoImage(image=img_resize)
    label_show_input_image.configure(image=cover)
    label_show_input_image.image = cover

    var_input_image_shape.set("(" + str(input_img_array.shape[0]) + ", " + str(input_img_array.shape[1]) + ")")


def show_query_img():
    global query_img_array
    w_box = 200
    h_box = 200
    img_path = askopenfilename()
    query_img = Image.open(img_path)
    query_img_array = np.asarray(query_img)

    w, h = query_img.size
    f1 = 1.0 * w_box / w
    f2 = 1.0 * h_box / h
    factor = min([f1, f2])
    resize_w = int(factor * w)
    resize_h = int(factor * h)
    img_resize = query_img.resize((resize_w, resize_h), Image.ANTIALIAS)

    cover = ImageTk.PhotoImage(image=img_resize)
    label_show_query_image.configure(image=cover)
    label_show_query_image.image = cover

    var_template_shape.set("(" + str(query_img_array.shape[0]) + ", " + str(query_img_array.shape[1]) + ")")


def show_template_match_result():
    if input_img_array is None or query_img_array is None:
        print("Either input image or query image is not open !")
        return 0
    global match_array
    global match_coor
    h, w = query_img_array.shape[0], query_img_array.shape[1]
    match_result = cv2.matchTemplate(input_img_array, query_img_array, cv2.TM_CCOEFF_NORMED)
    match_coor = np.array(np.unravel_index(np.argmax(match_result), match_result.shape))
    match_image = input_img_array.copy()
    match_image = cv2.rectangle(match_image, (match_coor[1], match_coor[0]), (match_coor[1] + w, match_coor[0] + h), (0, 255, 150), 5)
    match_array = input_img_array[match_coor[0]:match_coor[0] + h, match_coor[1]:match_coor[1] + w].copy()

    w_box = 400
    h_box = 400
    input_img = Image.fromarray(match_image)

    w, h = input_img.size
    f1 = 1.0 * w_box / w
    f2 = 1.0 * h_box / h
    factor = min([f1, f2])
    resize_w = int(factor * w)
    resize_h = int(factor * h)
    img_resize = input_img.resize((resize_w, resize_h), Image.ANTIALIAS)

    cover = ImageTk.PhotoImage(image=img_resize)
    label_show_input_image.configure(image=cover)
    label_show_input_image.image = cover

    button_show_diff.configure(state=tkinter.NORMAL)
    button_show_blurred_diff.configure(state=tkinter.NORMAL)

    var_coordinate.set("(" + str(match_coor[0]) + ", " + str(match_coor[1]) + ")")


def show_original_diff_result():
    global diff_original 
    diff_original = (query_img_array - match_array + 127).astype(np.uint8)
    w_box = 200
    h_box = 200

    diff_img_original = Image.fromarray(diff_original)

    w, h = diff_img_original.size
    f1 = 1.0 * w_box / w
    f2 = 1.0 * h_box / h
    factor = min([f1, f2])
    resize_w = int(factor * w)
    resize_h = int(factor * h)
    img_resize = diff_img_original.resize((resize_w, resize_h), Image.ANTIALIAS)

    cover = ImageTk.PhotoImage(image=img_resize)
    label_show_diff_image.configure(image=cover)
    label_show_diff_image.image = cover

    if match_coor[1] + match_array.shape[1] < input_img_array.shape[1]:
        button_x1.configure(state=tkinter.NORMAL)
    if match_coor[1] >= 1:
        button_x2.configure(state=tkinter.NORMAL)
    if match_coor[0] + match_array.shape[0] < input_img_array.shape[0]:
        button_y1.configure(state=tkinter.NORMAL)
    if match_coor[0] >= 1:
        button_y2.configure(state=tkinter.NORMAL)


def x_plus():
    match_coor[1:2] += 1
    if match_coor[1] + match_array.shape[1] >= input_img_array.shape[1]:
        button_x1.configure(state=tkinter.DISABLED)

    h, w = query_img_array.shape[0], query_img_array.shape[1]
    draw_and_show_rectangle(input_img_array, query_img_array, match_coor)

    offset_match_array = input_img_array[match_coor[0]:match_coor[0] + h, match_coor[1]:match_coor[1] + w].copy()
    show_diff_result(query_img_array, offset_match_array)

    if match_coor[1] + match_array.shape[1] < input_img_array.shape[1]:
        button_x1.configure(state=tkinter.NORMAL)
    if match_coor[1] >= 1:
        button_x2.configure(state=tkinter.NORMAL)
    if match_coor[0] + match_array.shape[0] < input_img_array.shape[0]:
        button_y1.configure(state=tkinter.NORMAL)
    if match_coor[0] >= 1:
        button_y2.configure(state=tkinter.NORMAL)

    var_coordinate.set("(" + str(match_coor[0]) + ", " + str(match_coor[1]) + ")")
    print(var_x_shift.get())
    print(var_y_shift.get())
    var_x_shift.set(var_x_shift.get() + 1)
    print(var_x_shift.get())
    print(var_y_shift.get())
    x_shift = var_x_shift.get()
    y_shift = var_y_shift.get()
    var_coordinate_shift.set("x: " + str(x_shift) + ", y: " + str(y_shift))


def x_minus():
    match_coor[1:2] -= 1
    if match_coor[1] <= 0:
        button_x2.configure(state=tkinter.DISABLED)

    h, w = query_img_array.shape[0], query_img_array.shape[1]
    draw_and_show_rectangle(input_img_array, query_img_array, match_coor)

    offset_match_array = input_img_array[match_coor[0]:match_coor[0] + h, match_coor[1]:match_coor[1] + w].copy()
    show_diff_result(query_img_array, offset_match_array)

    if match_coor[1] + match_array.shape[1] < input_img_array.shape[1]:
        button_x1.configure(state=tkinter.NORMAL)
    if match_coor[1] >= 1:
        button_x2.configure(state=tkinter.NORMAL)
    if match_coor[0] + match_array.shape[0] < input_img_array.shape[0]:
        button_y1.configure(state=tkinter.NORMAL)
    if match_coor[0] >= 1:
        button_y2.configure(state=tkinter.NORMAL)

    var_coordinate.set("(" + str(match_coor[0]) + ", " + str(match_coor[1]) + ")")
    var_x_shift.set(var_x_shift.get() - 1)
    x_shift = var_x_shift.get()
    y_shift = var_y_shift.get()
    var_coordinate_shift.set("x: " + str(x_shift) + ", y: " + str(y_shift))


def y_plus():
    match_coor[0:1] += 1
    if match_coor[0] + match_array.shape[0] >= input_img_array.shape[0]:
        button_y1.configure(state=tkinter.DISABLED)

    h, w = query_img_array.shape[0], query_img_array.shape[1]
    draw_and_show_rectangle(input_img_array, query_img_array, match_coor)

    offset_match_array = input_img_array[match_coor[0]:match_coor[0] + h, match_coor[1]:match_coor[1] + w].copy()
    show_diff_result(query_img_array, offset_match_array)

    if match_coor[1] + match_array.shape[1] < input_img_array.shape[1]:
        button_x1.configure(state=tkinter.NORMAL)
    if match_coor[1] >= 1:
        button_x2.configure(state=tkinter.NORMAL)
    if match_coor[0] + match_array.shape[0] < input_img_array.shape[0]:
        button_y1.configure(state=tkinter.NORMAL)
    if match_coor[0] >= 1:
        button_y2.configure(state=tkinter.NORMAL)

    var_coordinate.set("(" + str(match_coor[0]) + ", " + str(match_coor[1]) + ")")
    var_y_shift.set(var_y_shift.get() + 1)
    x_shift = var_x_shift.get()
    y_shift = var_y_shift.get()
    var_coordinate_shift.set("x: " + str(x_shift) + ", y: " + str(y_shift))


def y_minus():
    match_coor[0:1] -= 1
    if match_coor[0] <= 0:
        button_x2.configure(state=tkinter.DISABLED)

    h, w = query_img_array.shape[0], query_img_array.shape[1]
    draw_and_show_rectangle(input_img_array, query_img_array, match_coor)

    offset_match_array = input_img_array[match_coor[0]:match_coor[0] + h, match_coor[1]:match_coor[1] + w].copy()
    show_diff_result(query_img_array, offset_match_array)

    if match_coor[1] + match_array.shape[1] < input_img_array.shape[1]:
        button_x1.configure(state=tkinter.NORMAL)
    if match_coor[1] >= 1:
        button_x2.configure(state=tkinter.NORMAL)
    if match_coor[0] + match_array.shape[0] < input_img_array.shape[0]:
        button_y1.configure(state=tkinter.NORMAL)
    if match_coor[0] >= 1:
        button_y2.configure(state=tkinter.NORMAL)

    var_coordinate.set("(" + str(match_coor[0]) + ", " + str(match_coor[1]) + ")")
    var_y_shift.set(var_y_shift.get() - 1)
    x_shift = var_x_shift.get()
    y_shift = var_y_shift.get()
    var_coordinate_shift.set("x: " + str(x_shift) + ", y: " + str(y_shift))


root = tkinter.Tk()
root.title("Honghua template matching")

label_input = tkinter.Label(root, text="Input image: ")
label_input.grid(row=0, column=0, sticky=tkinter.W, padx=5)


frame_input = tkinter.Frame(root, height=400, width=400, relief='raised')
# fix the frame to a fixed size (400 * 400)
frame_input.pack_propagate(0)
frame_input.grid(row=1, column=0, padx=5)
label_show_input_image = tkinter.Label(frame_input)
label_show_input_image.pack(fill=tkinter.BOTH)

label_query = tkinter.Label(root, text="Query image: ")
label_query.grid(row=2, column=0, sticky=tkinter.W, padx=5)
frame_query = tkinter.Frame(root, height=200, width=200)
frame_query.pack_propagate(0)
frame_query.grid(row=3, column=0, padx=5)
label_show_query_image = tkinter.Label(frame_query)
label_show_query_image.pack(fill=tkinter.BOTH)

label_diff_result = tkinter.Label(root, text="Diff result: ")
label_diff_result.grid(row=2, column=1, sticky=tkinter.W, padx=5)
frame_diff = tkinter.Frame(root, height=200, width=200)
frame_diff.pack_propagate(0)
frame_diff.grid(row=3, column=1, padx=5)
label_show_diff_image = tkinter.Label(frame_diff)
label_show_diff_image.pack(fill=tkinter.BOTH)

label_blurred_diff_result = tkinter.Label(root, text="Blurred diff result: ")
label_blurred_diff_result.grid(row=2, column=2, sticky=tkinter.W, padx=5)
frame_blurred_diff = tkinter.Frame(root, height=200, width=200)
frame_blurred_diff.pack_propagate(0)
frame_blurred_diff.grid(row=3, column=2, padx=5)
label_show_blurred_diff_image = tkinter.Label(frame_blurred_diff)
label_show_blurred_diff_image.pack(fill=tkinter.BOTH)


Button_group = tkinter.Frame(root, relief='raised')
Button_group.grid(row=4, column=0, columnspan=3, pady=5)
button_open_1 = tkinter.Button(Button_group, text='open1', command=show_input_img)
button_open_1.grid(row=0, column=0, sticky=tkinter.W)
button_open_2 = tkinter.Button(Button_group, text='open2', command=show_query_img)
button_open_2.grid(row=0, column=1)
button_match = tkinter.Button(Button_group, text='match', command=show_template_match_result)
button_match.grid(row=0, column=2)
button_show_diff = tkinter.Button(Button_group, text='diff', command=show_original_diff_result, state=tkinter.DISABLED)
button_show_diff.grid(row=0, column=3)
button_x1 = tkinter.Button(Button_group, text='x+', state=tkinter.DISABLED, command=x_plus)
button_x1.grid(row=0, column=4)
button_x2 = tkinter.Button(Button_group, text='x-', state=tkinter.DISABLED, command=x_minus)
button_x2.grid(row=0, column=5)
button_y1 = tkinter.Button(Button_group, text='y+', state=tkinter.DISABLED, command=y_plus)
button_y1.grid(row=0, column=6)
button_y2 = tkinter.Button(Button_group, text='y-', state=tkinter.DISABLED, command=y_minus)
button_y2.grid(row=0, column=7)
button_show_blurred_diff = tkinter.Button(Button_group, text='blurred diff', state=tkinter.DISABLED, command=show_blurred_diff_result)
button_show_blurred_diff.grid(row=0, column=8)
button_save_diff_res = tkinter.Button(Button_group, text='save_diff', command=save_diff)
button_save_diff_res.grid(row=0, column=9)


var_coordinate = tkinter.StringVar()
var_input_image_shape = tkinter.StringVar()
var_template_shape = tkinter.StringVar()
var_coordinate_shift = tkinter.StringVar()
var_x_shift = tkinter.IntVar()
var_y_shift = tkinter.IntVar()

var_coordinate.set("(None, None)")
var_input_image_shape.set("(None, None)")
var_template_shape.set("(None, None)")
var_coordinate_shift.set("x: 0, y: 0")
var_x_shift.set(0)
var_y_shift.set(0)

info_panel = tkinter.Frame(root)
info_panel.grid(row=0, column=1, rowspan=2, columnspan=2, padx=5, pady=5, sticky=tkinter.NW)
label_image_info_1 = tkinter.Label(info_panel, text='Input image shape: ')
label_image_info_1.pack(side='top', anchor='w')
label_image_info_2 = tkinter.Label(info_panel, textvariable=var_input_image_shape)
label_image_info_2.pack(side='top', anchor='w')
label_template_info_1 = tkinter.Label(info_panel, text="Template shape: ")
label_template_info_1.pack(side='top', anchor='w')
label_template_info_2 = tkinter.Label(info_panel, textvariable=var_template_shape)
label_template_info_2.pack(side='top', anchor='w')
label_match_coor_1 = tkinter.Label(info_panel, text="Matching coordinates: ")
label_match_coor_1.pack(side='top', anchor='w')
label_match_coor_2 = tkinter.Label(info_panel, textvariable=var_coordinate)
label_match_coor_2.pack(side='top', anchor='w')
label_shift_coor_1 = tkinter.Label(info_panel, text="Shift of coordinates: ")
label_shift_coor_1.pack(side='top', anchor='w')
label_shift_coor_2 = tkinter.Label(info_panel, textvariable=var_coordinate_shift)
label_shift_coor_2.pack(side='top', anchor='w')
label_save_diff = tkinter.Label(info_panel, text="save name for diff image")
label_save_diff.pack(side='top', anchor='w')
save_file_name_entry = tkinter.Entry(info_panel)
save_file_name_entry.pack(side='top', anchor='w')

tkinter.mainloop()
