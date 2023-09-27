import mysql.connector
from keyboard import read_event
import keyboard

connection = mysql.connector.connect(
    host="localhost",
    user="product",
    password="zxcasdqwe113",
    db="project_product"
)

cursor = connection.cursor()

try:
    barcode_data = ""
    while True:
        event = keyboard.read_event(suppress=True)

        if event.event_type == 'down':
            if event.name == 'enter':
                if barcode_data:
                    barcode_query = "SELECT * FROM add_product WHERE BarcodeKey=%s"
                    cursor.execute(barcode_query, (barcode_data,))
                    barcodekey = cursor.fetchall()
                    print("Barcode :", barcode_data)
                    if barcodekey:
                        print("BarcodeKey Found:", barcodekey)

                    # อ่านผลลัพธ์จากคำสั่ง SQL ก่อนหน้านี้และปิดผลลัพธ์
                    cursor.fetchall()
                    cursor.close()
                    cursor = connection.cursor()

                barcode_data = ""
                break
            else:
                barcode_data += event.name

except KeyboardInterrupt:
    pass

cursor.close()
connection.close()

