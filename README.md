# Canny Edge 추출 실습 보고서

## 1. 개요
본 보고서는 딥러닝 과제 #1의 일환으로, Canny Edge 추출 알고리즘을 얼굴 이미지에 적용한 결과를 설명합니다. OpenCV 라이브러리를 활용하여 이미지의 윤곽선을 효과적으로 검출하는 과정을 구현했습니다.

## 2. 이론적 배경
Canny Edge 검출 알고리즘은 John F. Canny가 1986년에 개발한 알고리즘으로, 다음과 같은 주요 단계로 구성됩니다:

1. **노이즈 제거**: 가우시안 필터를 사용하여 이미지의 노이즈를 제거합니다.
2. **그래디언트 계산**: Sobel 필터를 사용하여 이미지의 x, y 방향 그래디언트를 계산합니다.
3. **비최대 억제(Non-maximum Suppression)**: 에지의 두께를 1픽셀로 줄입니다.
4. **이중 임계값 처리**: 강한 에지와 약한 에지를 구분합니다.
5. **히스테리시스를 통한 에지 추적**: 강한 에지와 연결된 약한 에지만 최종 에지로 선택합니다.

이러한 과정을 통해 노이즈에 강하고 정확한 윤곽선 검출이 가능합니다.

## 3. 구현 코드 설명

코드는 Google Colab 환경에서 작성되었으며, 다음과 같은 단계로 구성됩니다:

1. 필요한 라이브러리 임포트 (OpenCV, NumPy, Matplotlib)
2. 사용자로부터 이미지 업로드 받기
3. 이미지를 그레이스케일로 변환
4. Canny Edge 검출 알고리즘 적용
5. 원본, 그레이스케일, 에지 검출 결과를 시각적으로 비교

```python
# 필요한 라이브러리 임포트
from google.colab import files
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# 이미지 업로드
uploaded = files.upload()
file_name = list(uploaded.keys())[0]

# 이미지 읽기 및 변환
img = cv.imread(file_name)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Canny Edge 검출 적용
# 첫 번째 임계값(100)과 두 번째 임계값(200)을 사용
canny_edge = cv.Canny(gray, 100, 200)

# 시각화를 위한 색상 변환 (BGR -> RGB)
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# 결과 시각화
plt.figure(figsize=(15,5))

# 원본 이미지
plt.subplot(1, 3, 1)
plt.title("Original")
plt.imshow(img_rgb)
plt.axis('off')

# 그레이스케일 이미지
plt.subplot(1, 3, 2)
plt.title("Grayscale")
plt.imshow(gray, cmap='gray')
plt.axis('off')

# Canny Edge 검출 결과
plt.subplot(1, 3, 3)
plt.title("Canny Edge")
plt.imshow(canny_edge, cmap='gray')
plt.axis('off')

plt.show()
```

## 4. 코드 분석

### 주요 함수 설명
- `cv.cvtColor(img, cv.COLOR_BGR2GRAY)`: 이미지를 그레이스케일로 변환합니다. 에지 검출은 일반적으로 컬러 정보가 아닌 밝기 정보에 기반하기 때문입니다.
- `cv.Canny(gray, 100, 200)`: Canny 알고리즘을 적용합니다. 두 개의 임계값(100, 200)을 사용합니다:
  - 첫 번째 임계값(100): 이 값보다 낮은 그래디언트는 에지로 고려되지 않습니다.
  - 두 번째 임계값(200): 이 값보다 높은 그래디언트는 강한 에지로 간주됩니다.
  - 첫 번째와 두 번째 임계값 사이의 그래디언트는 강한 에지와 연결된 경우에만 에지로 간주됩니다.

### 매개변수 선택
임계값 100과 200은 실험적으로 선택되었으며, 얼굴 이미지의 주요 윤곽선을 효과적으로 검출하는 균형점입니다. 더 낮은 임계값은 더 많은 에지를 검출하지만 노이즈도 함께 검출될 수 있고, 더 높은 임계값은 중요한 에지를 놓칠 수 있습니다.

## 5. 실험 결과
얼굴 이미지에 Canny Edge 검출 알고리즘을 적용한 결과:
- 주요 얼굴 특징(눈, 코, 입, 얼굴 윤곽선)이 명확하게 검출되었습니다.
- 그레이스케일 변환을 통해 이미지의 텍스처와 명암 차이가 보존되었습니다.
- 임계값 조정을 통해 중요한 에지만 검출되고 불필요한 세부사항은 제거되었습니다.

## 6. 결론
Canny Edge 검출 알고리즘은 얼굴 이미지의 주요 특징을 효과적으로 검출했습니다. 이 알고리즘은 컴퓨터 비전 분야에서 기초적이지만 강력한 도구로, 얼굴 인식, 객체 검출 등의 더 복잡한 작업을 위한 전처리 단계로 활용될 수 있습니다.

본 실습을 통해 이미지 처리의 기본 개념과 OpenCV 라이브러리의 활용 방법을 효과적으로 학습할 수 있었습니다.
