import cv2
import numpy as np
def overlay(image, mask, color, alpha, resize=None):
    color = color[::-1]
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    colored_mask = np.moveaxis(colored_mask, 0, -1)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
    image_overlay = masked.filled()

    if resize is not None:
        image = cv2.resize(image.transpose(1, 2, 0), resize)
        image_overlay = cv2.resize(image_overlay.transpose(1, 2, 0), resize)

    image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)

    return image_combined
# Load the image
image = cv2.imread('/home/baoanh/baoanh/DATN/ESFPNet/Endoscope-WL/UTDD_Splited/testSplited/images/242.jpeg')

# Load the segment mask (make sure it's a binary mask)
segment_mask = cv2.imread('/home/baoanh/baoanh/DATN/dataset/Ungthudaday/test/mask_images/242.png', 0)
print(segment_mask.shape)
image_with_masks = overlay(image, segment_mask, color=(0,255,0), alpha=0.3)
# # Find the contours of the segmented region
# contours, _ = cv2.findContours(segment_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# print(np.unique(segment_mask), segment_mask.shape, image.shape, type(segment_mask))
# # Draw boundaries around the segmented region
# for contour in contours:
#     cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)  # Draw a green boundary (color: (0, 255, 0), thickness: 2)

# # Save or display the image with the boundaries
# cv2.imwrite('output_image.jpg', image)  # Save the image with boundaries
cv2.imshow('Segmented Image', image_with_masks)  # Display the image with boundaries

# Close the display window when a key is pressed
cv2.waitKey(0)
cv2.destroyAllWindows()
