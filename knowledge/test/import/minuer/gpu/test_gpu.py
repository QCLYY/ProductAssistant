#一 验证GPU加速

import torch
print(torch.__version__)
print(f"是否支持GPU:{torch.cuda.is_available()}")
print(f"设备名{torch.cuda.get_device_name()}")


#二   配置模型的下载地址（下载本地模型）
#1.默认找GPU加速
#2.默认把用到和解析相关的模型都缓存到C盘用户目录的.cache中
    #2.1修改模型目录，执行命令之前通过环境变量(HF_HOME/MODELSCOPE_CACHE)设置 $env:MODELSCOPE_CACHE="E:\AI+Py\ai_models"
    #2.2下载模型 mineru-models-download  会自动生成mineru.json
#3.默认使用解析后端是混合模式（hybrid:pineline+vlm）
#mineru-models-download

#三 指定使用本地模型进行解析
#mineru -p <input_path> -o <output_path> --source local
#mineru -p E:\\AI+Py\\shopkeeper_brain\\knowledge\\processor\\import_process\\import_temp_dir\\万用表RS-12的使用.pdf -o E:\\AI+Py\\shopkeeper_brain\\knowledge\\processor\\import_process\\output_temp_dir --source local