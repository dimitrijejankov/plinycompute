import numpy as np
import cv2

cap = cv2.VideoCapture('/home/dimitrije/PycharmProjects/gen/chaplin.mp4')

# how many times we split over the width
x_split = 8

# how many times we split over the height
y_split = 8

# we are going to have three blocks along z direction
z_split = 3

# how many frames are in z
z_count = 50

stack = []


def write_out_chunks(out, f):

    chunk_x = out.shape[1] // 8
    chunk_y = out.shape[2] // 8

    for x in range(x_split):
        for y in range(y_split):
            for c in range(out.shape[3]):

                # write the channel to bytes
                tmp = out[:, x * chunk_x:(x + 1) * chunk_x, y * chunk_y:(y + 1) * chunk_y, c]
                f.write(tmp.tobytes())

wrote_header = False

# open the file
f = open('./out.bin', 'wb')

# the current z frame in the block
current_z = 0

# while we have frames do this
while True:

    # read the frame
    ret, frame = cap.read()

    # do we have a frame?
    if frame is not None:

        # do we need to write out the header
        if not wrote_header:

            # write the header
            f.write(x_split.to_bytes(4, byteorder='little'))
            f.write(y_split.to_bytes(4, byteorder='little'))
            f.write(z_split.to_bytes(4, byteorder='little'))
            f.write((frame.shape[0] // 8).to_bytes(4, byteorder='little'))
            f.write((frame.shape[1] // 8).to_bytes(4, byteorder='little'))
            f.write(z_count.to_bytes(4, byteorder='little'))

            # mark that we have wrote to the header
            wrote_header = True

        # increment the z
        current_z += 1

        # append the frame
        stack.append(frame)

        # if we have enough frames
        if current_z == 50:

            # combine all the frames
            out = np.stack(stack, axis=0)
            stack.clear()

            # write the chunks out
            write_out_chunks(out, f)

            # reset the counter
            current_z = 0

    # get the frame
    elif frame is None:
        break

f.close()

cap.release()
cv2.destroyAllWindows()
