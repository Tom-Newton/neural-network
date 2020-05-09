with open('optdigits-orig.cv', 'r') as file:
    data = file.readlines()
# Remove header
data = data[21:]

image_height = 32
with open('optdigits-formatted.tes', 'w+') as file:
    n = 0
    while n < len(data):
        line = ''
        for i in range(image_height):
            for character in data[n + i][:-1]:
                line += character
                line += ','
        line += data[n + image_height][1]
        line += '\n'
        file.write(line)
        n += (image_height + 1)
