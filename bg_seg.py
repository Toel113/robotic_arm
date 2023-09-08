import cv2

# Load a video file
cap = cv2.VideoCapture(0)

# Create a background subtractor object
bg_subtractor = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Apply background subtraction
    fg_mask = bg_subtractor.apply(frame)

    # Optionally, you can further process the mask, like applying morphology operations to clean it up.

    # Display the original frame and foreground mask
    #cv2.circle(frame,(200,200), 10, (0,0,255), 2)
    #cv2.imshow('Original', frame)
    cv2.imshow('Foreground Mask', fg_mask)

    if cv2.waitKey(1) == ord('q'):  # Press 'Esc' key to exit
        break

cap.release()
cv2.destroyAllWindows()

