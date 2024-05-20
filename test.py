import pickle
import numpy as np


# # 加载 g.npy 文件
# g_loaded = np.load('g.npy')
# print("\ng:")
# print(len(g_loaded))

# with open('predictions.pkl', 'rb') as f:
#     q_image_ids_loaded = pickle.load(f)

# # 打印加载的对象

# print(len(q_image_ids_loaded
#           ))
# concatenated_array = np.concatenate(q_image_ids_loaded, axis=0)

# # 打印连接后的数组
# print("Concatenated array shape:", concatenated_array.shape)
# print(concatenated_array[0])
# with open('q_image_ids_cat.pkl', 'wb') as f:
#     q_image_ids_loaded = pickle.dump(concatenated_array,f)
from tqdm import tqdm
with open('q_image_ids_cat.pkl', 'rb') as f:
    q_image_ids = pickle.load(f)
with open('g_image_ids_cat.pkl', 'rb') as f:
    g_image_ids = pickle.load(f)
with open('predictions.pkl', 'rb') as f:
    predictions = pickle.load(f)
print('ID:',q_image_ids[0],'\npred:',predictions[0][:6])
with open('submit.csv', 'w') as f:
    f.write('Id,Predicted\n')
    for i, q_id in tqdm(enumerate(q_image_ids)):
        predicted_ids = predictions[i]  # 假设 predictions 是相似图像的索引
        predicted_ids_str = ' '.join(map(str, predicted_ids))
        predicted_image_ids = [g_image_ids[idx][:-4] for idx in predicted_ids]  # 根据索引找到对应的图像ID
        predicted_image_ids_str = ' '.join(map(str, predicted_image_ids))
        f.write(f'{q_id},{predicted_image_ids_str}\n')
