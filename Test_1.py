import cv2
import time

# Morse code dictionary
MORSE_DICT = {
    '.-': 'A', '-...': 'B', '-.-.': 'C', '-..': 'D',
    '.': 'E', '..-.': 'F', '--.': 'G', '....': 'H',
    '..': 'I', '.---': 'J', '-.-': 'K', '.-..': 'L',
    '--': 'M', '-.': 'N', '---': 'O', '.--.': 'P',
    '--.-': 'Q', '.-.': 'R', '...': 'S', '-': 'T',
    '..-': 'U', '...-': 'V', '.--': 'W', '-..-': 'X',
    '-.--': 'Y', '--..': 'Z', '': ' '
}

# Fine-tuned thresholds for 300ms Morse units
BRIGHTNESS_THRESHOLD = 140
DOT_THRESHOLD = 0.5
LETTER_GAP = 0.8
WORD_GAP = 1.8

# State variables
light_on = False
current_symbol = ''
decoded_text = ''
on_start_time = None
off_start_time = time.time()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detection region
    h, w = gray.shape
    region = gray[h//2 - 80:h//2 + 80, w//2 - 80:w//2 + 80]
    brightness = region.mean()

    # Draw region box
    cv2.rectangle(frame, (w//2 - 80, h//2 - 80), (w//2 + 80, h//2 + 80), (255, 0, 0), 2)

    current_time = time.time()

    if brightness > BRIGHTNESS_THRESHOLD:
        if not light_on:
            light_on = True
            on_start_time = current_time

            gap = current_time - off_start_time
            if gap >= WORD_GAP:
                letter = MORSE_DICT.get(current_symbol, '')
                decoded_text += letter + ' '
                current_symbol = ''
            elif gap >= LETTER_GAP:
                letter = MORSE_DICT.get(current_symbol, '')
                decoded_text += letter
                current_symbol = ''
    else:
        if light_on:
            light_on = False
            off_start_time = current_time
            duration = current_time - on_start_time

            if duration <= DOT_THRESHOLD:
                current_symbol += '.'
            else:
                current_symbol += '-'

    # Display everything
    cv2.putText(frame, f"Morse: {current_symbol}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"Text: {decoded_text}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(frame, f"Brightness: {int(brightness)}", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)
    cv2.putText(frame, "Press 'C'=clear, 'Space'=flush, 'Q'=quit", (10, 460),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow("Morse Code Torch Decoder", frame)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('c'):
        current_symbol = ''
        decoded_text = ''
    elif key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord(' '):  # Spacebar to flush
        if current_symbol:
            letter = MORSE_DICT.get(current_symbol, '')
            decoded_text += letter
            current_symbol = ''

cap.release()
cv2.destroyAllWindows()
