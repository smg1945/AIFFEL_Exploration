# AIFFEL Campus Online 4th Code Peer Review Templete
- 코더 : 이하영
- 리뷰어 : 이효준

<aside>
🔑 **PRT(Peer Review Template)**

- [x]  1.코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
| 네, 잘 해결하였습니다.
- [x]  2.주석을 보고 작성자의 코드가 이해되었나요?
```python
image = copy.deepcopy(item['image'])
image = cv2.drawContours(image, [rect], 0, (0,0,255), 2)
image_grad = cv2.drawContours(image, [rect_grad], 0, (0,0,255), 2)
plt.imshow(image)
plt.imshow(image_grad)
plt.show()
```
| CAM을 통해 바운딩 박스를 손쉽게 그리는 부분이 잘 이해됐습니다.  
| 이후 IoU 계산과 짧은 회고가 인상적이었습니다.
    
- [ ]  3.코드가 에러를 유발할 가능성이 있나요?
    
| 에러를 유발하는 부분은 보이지 않았습니다.
    
- [x]  4.코드 작성자가 코드를 제대로 이해하고 작성했나요? (직접 인터뷰해보기)
```python
plt.subplot(2, 2, 1)
plt.plot(history_cam_model.history['loss'], 'r')
plt.plot(history_cam_model.history['val_loss'], 'b')
plt.legend(['train_loss', 'val_loss'], loc='upper left')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(history_cam_model.history['accuracy'], 'r')
plt.plot(history_cam_model.history['val_accuracy'], 'b')
plt.legend(['train_accuracy', 'val_accuracry'], loc='upper left')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.grid(True)

plt.tight_layout()
plt.show()

print('train_loss:', history_cam_model.history['loss'][-1], 'val_loss:', history_cam_model.history['val_loss'][-1])
print('train_accuracy:', history_cam_model.history['accuracy'][-1], 'val_accuracy:', history_cam_model.history['val_accuracy'][-1])
```
| 모델 학습 기록을 시각화하여 cam_model이 잘 학습되고 있음을 확인할 수 있었다.
    
- [x]  5.코드가 간결한가요?
```python
def get_iou(gt_bbox, pred_bbox):
    y_min = max(gt_bbox[0], pred_bbox[0])
    x_min = max(gt_bbox[1], pred_bbox[1])
    y_max = min(gt_bbox[2], pred_bbox[2])
    x_max = min(gt_bbox[3], pred_bbox[3])

    interArea = max(0, x_max - x_min) * max(0, y_max - y_min)
    gt_bbox_area = (gt_bbox[2] - gt_bbox[0]) * (gt_bbox[3] - gt_bbox[1])
    pred_bbox_area = (pred_bbox[2] - pred_bbox[0]) * (pred_bbox[3] - pred_bbox[1])
    
    iou = interArea / float(gt_bbox_area + pred_bbox_area - interArea)

    return iou
```
| 간결하게 IoU를 계산하는 함수로 보기 좋았습니다.
    
</aside>
