import numpy as np
import nibabel as nib
import os
from pathlib import Path

def convert_single_npz(npz_path, output_path, spacing=(0.7, 0.7, 1.0)):
    """
    转换单个 NPZ 文件为 NIFTI 格式
    """
    try:
        # 加载 NPZ 文件
        data = np.load(npz_path)
        print(f"\n处理文件: {npz_path.name}")
        print(f"包含的数组: {list(data.keys())}")
        
        # 获取图像数据 (所有文件都包含 'data' 键)
        image_data = data['data']
        key_used = 'data'
        
        print(f"使用数组 '{key_used}': 形状={image_data.shape}, 类型={image_data.dtype}")
        print(f"数值范围: [{image_data.min():.2f}, {image_data.max():.2f}]")
        
        # 处理4D数据（如果存在）
        if image_data.ndim == 4:
            print("检测到4D数据，取第一个通道")
            image_data = image_data[0] if image_data.shape[0] < image_data.shape[-1] else image_data[..., 0]
        
        # 针对标注数据，转换为uint8类型
        if 'artery' in str(npz_path) or 'vein' in str(npz_path):
            # 标注数据：二值化掩码
            image_data = image_data.astype(np.uint8)
            print("标注数据已转换为uint8格式")
        else:
            # CT数据：保持int16格式
            image_data = image_data.astype(np.int16)
            print("CT数据保持int16格式")
        
        # 创建仿射矩阵
        affine = np.eye(4)
        affine[0, 0] = spacing[0]
        affine[1, 1] = spacing[1]
        affine[2, 2] = spacing[2]
        
        # 创建并保存NIFTI文件
        nifti_img = nib.Nifti1Image(image_data, affine)
        nib.save(nifti_img, output_path)
        print(f"✓ 成功保存: {output_path.name}")
        
        return True
        
    except Exception as e:
        print(f"✗ 转换失败 {npz_path.name}: {str(e)}")
        return False

def convert_5_samples():
    """
    转换前5个样本
    """
    base_path = Path("D:/yangmiaomiao/study/ucl/final/3D_lung_segmentation/dataset")
    output_base = Path("D:/yangmiaomiao/study/ucl/final/3D_lung_segmentation/dataset/converted_samples")
    
    # 创建输出目录
    (output_base / "ct_scan").mkdir(parents=True, exist_ok=True)
    (output_base / "artery").mkdir(parents=True, exist_ok=True)
    (output_base / "vein").mkdir(parents=True, exist_ok=True)
    
    # 要转换的文件编号
    sample_ids = ["001", "002", "003", "004", "005"]
    
    print("开始转换5个样本...")
    print("=" * 50)
    
    for sample_id in sample_ids:
        print(f"\n🔄 处理样本 {sample_id}")
        
        # CT扫描文件
        ct_file = base_path / "ct_scan" / f"{sample_id}.npz"
        if ct_file.exists():
            ct_output = output_base / "ct_scan" / f"{sample_id}.nii.gz"
            convert_single_npz(ct_file, ct_output, spacing=(0.7, 0.7, 1.0))
        else:
            print(f"⚠️  CT文件不存在: {ct_file}")
        
        # 动脉标注文件
        artery_file = base_path / "annotation" / "artery" / f"{sample_id}.npz"
        if artery_file.exists():
            artery_output = output_base / "artery" / f"{sample_id}.nii.gz"
            convert_single_npz(artery_file, artery_output, spacing=(0.7, 0.7, 1.0))
        else:
            print(f"⚠️  动脉文件不存在: {artery_file}")
        
        # 静脉标注文件
        vein_file = base_path / "annotation" / "vein" / f"{sample_id}.npz"
        if vein_file.exists():
            vein_output = output_base / "vein" / f"{sample_id}.nii.gz"
            convert_single_npz(vein_file, vein_output, spacing=(0.7, 0.7, 1.0))
        else:
            print(f"⚠️  静脉文件不存在: {vein_file}")
    
    print("\n" + "=" * 50)
    print("转换完成！")
    print(f"输出目录: {output_base}")
    print("\n📋 在3D Slicer中加载步骤：")
    print("1. 打开3D Slicer")
    print("2. File → Add Data")
    print("3. 选择转换后的 .nii.gz 文件")
    print("4. CT数据作为背景，血管标注作为前景或分割显示")

if __name__ == "__main__":
    convert_5_samples()