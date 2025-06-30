import pydicom
import os
import numpy as np
from PIL import Image
from matplotlib import pyplot
import pandas as pd


file_name = "IMG0002.dcm"
file_name_ct = "IMG0001.dcm"

def read_dcm():
    
    ds = pydicom.dcmread(file_name)
    ds_ct = pydicom.dcmread(file_name_ct)
    
    print(ds.PatientID, ds.StudyDate, ds.Modality)
    
    data = np.array(ds.pixel_array)
    data_ct = np.array(ds_ct.pixel_array)
    data_img = Image.fromarray(ds.pixel_array)
    data_img_ct = Image.fromarray(ds_ct.pixel_array)
    data_img_rotated = data_img.rotate(angle=45, resample=Image.BICUBIC, fillcolor=data_img.getpixel((0,0)))
    print(data.dtype)
    data_rotated = np.array(data_img_rotated, dtype=np.int16)

    print(data_img)
    print(data_img_ct)

    pyplot.imshow(ds.pixel_array, cmap=pyplot.cm.bone)
    pyplot.show()
    pyplot.close() 


    print('ok')
    

def read_excel():
    file_path = 'ceshi.xlsx'
    # 读取xlsx文件
    data = pd.read_excel(file_path, sheet_name='Sheet2')

    # 显示数据
    # print(data)

    # 取第2列和第7列
    # data_new = data.iloc[:, [2, 7]]

    # 取第16至28行
    data_new = data.iloc[14:26, :]
    # 这里的data_new是一个DataFrame对象
    # 如果需要转换为numpy数组，可以使用values属性

    # data_new_df = data_new.iloc[14:16, :]
    data_new_df = data_new[['姓名', '肺结节', '肺结节部位', 'CT上界', 'CT下界', 'PET上界', 'PET下界']]

    print(data_new_df)
    
    filename = data_new_df.at[14, '姓名']
    filepath = '26-40/26-40/'
    img_path = os.path.join(filepath, filename, 'images')

    all_files = os.listdir(img_path)
    print(all_files)

    ct_path = os.path.join(img_path, all_files[1])
    ct_slices_files = os.listdir(ct_path)
    ct_slices = [os.path.join(ct_path, f) for f in ct_slices_files]
    ct_img = [pydicom.dcmread(f) for f in ct_slices]

    # 按照InstanceNumber排序
    ct_slices.sort(key=lambda x: int(x.InstanceNumber))
    # 提取像素数据
    ct_images = [s.pixel_array for s in ct_slices]
    # 将像素数据转换为numpy数组
    ct_images_np = np.array(ct_images)
    print(ct_images_np.shape)

    # 








    # data_new_np = data_new.values
    # print(data_new_np[0])
    # print(data_new_np[0][1])
    # print(data_new_np[0][5])
    # # 答应出第0行9-12列的值
    # print(data_new_np[0][9:12])




    print(data_new)

    print('!')





def main():
    read_excel()
    read_dcm()


if __name__ == "__main__":
    main()
