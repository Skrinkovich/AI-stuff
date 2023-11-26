

import os
import cv2 as cv


import face_recognition
from datetime import datetime, timedelta
import numpy as np
import platform
import pickle


# энкодинги (машиночитаемые обработанные алгоритмом) уже известные лица
known_face_encodings = []
# дополнительная информация к каждому лицу (имя, послений визит итд)
known_face_metadata = []

# сохраняем (сериализируем) списки энкодинга лиц и метадаты в файл.
def save_known_faces():
    with open("known_faces.dat", "wb") as face_data_file:
        face_data = [known_face_encodings, known_face_metadata]
        pickle.dump(face_data, face_data_file)
        print("Известные лица сохранены")

# загружаем (десериализуем) лица в программу при старте
def load_known_faces():
    global known_face_encodings, known_face_metadata
    try:
        with open("known_faces.dat", "rb") as face_data_file:
            known_face_encodings, known_face_metadata = pickle.load(face_data_file)
            print("Известные лица загружены")
    except FileNotFoundError as e:
        try:
            # если файла с лицами нет, то мы проходим по заготовленной папке с теми лицами, которые мы хотим показать алгоритму до работы и записываем их
            addKnownFacesFromFolder(folderName="knownFaces")
        except Exception as e:
            print("известных лиц нет в файле энкодингов и папка с заготовленными лицами пуста")
            pass

# для того, чтобы поток видео можно было передать в opencv из Jetson Nano к нему нужно добавить правильное описание
def get_jetson_gstreamer_source(capture_width=1280, capture_height=720, display_width=1280, display_height=720, framerate=60, flip_method=0):
    return (
            f'nvarguscamerasrc ! video/x-raw(memory:NVMM), ' +
            f'width=(int){capture_width}, height=(int){capture_height}, ' +
            f'format=(string)NV12, framerate=(fraction){framerate}/1 ! ' +
            f'nvvidconv flip-method={flip_method} ! ' +
            f'video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! ' +
            'videoconvert ! video/x-raw, format=(string)BGR ! appsink'
            )

# добавляем нового человека в список известных
def register_new_face(face_encoding, face_image, HumanName ="Unknown Human"):
    # добавляем новый энкодинг лица в список
    known_face_encodings.append(face_encoding)
    # в список метаданных добавляем в конец словарь с данными об этом человеке. Когда видели, сколько раз, фотку лица, имя.
    known_face_metadata.append({
        "first_seen": datetime.now(),
        "first_seen_this_interaction": datetime.now(),
        "last_seen": datetime.now(),
        "seen_count": 1,
        "seen_frames": 1,
        "face_image": face_image,
        "Name": HumanName,
    })

# если мы уже имеем лица, которые мы хотим добавить в качестве известных, то мы можем положить их в папку и эта функция сохраняет их до работы
def addKnownFacesFromFolder(folderName = "knownFaces"):
    folder1 = os.path.abspath(folderName)  # полный путь до папки с фотографиями
    listdir = os.listdir(folder1)  # список всех фотографий
    images = {} #  словарь заполним лицами, где ключ имя файла, значение- лицо. (можно назвать каждое лицо по имени)

    for each in listdir:
        images[each.split(".")[0]] = cv.imread(folder1 + "/" + each)#  добавляем фотографию лица в словаь

    for name, image in images.items():
        face_locations, face_encodings, face_labels = RetrieveFacesAndData(image) #  получаем местоположение каждого лица на фото, энкодинги лиц и подпись, но тут она не нужна
        # если лиц на фото будет несколько, то мы сохраним каждое из них
        for face_location, face_encoding in zip(face_locations, face_encodings):
            if lookup_known_face(face_encoding) == None:# проверка на то, что мы это лицо не знаем.
                register_new_face(face_encoding, image, HumanName = name)# добавляем в список новое лицо
                save_known_faces()# сохранияем список лиц в файл



# проверка на то, что мы уже знаем это лицо
def lookup_known_face(face_encoding):

    metadata = None

    # если список энкодингов лиц пустой, то функция возврщает None
    if len(known_face_encodings) == 0:
        return None

    # вызываем из библиотеки face_recognition функцию, которая проходит по списку энкодингов лиц и сравнивает полученный вариант с ними
    # возвращает она список расстояний для каждого лица от 0 до 1. Чем число меньше, тем похожесть больше
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

    # выбираем наименьшее значение из полученного списка
    best_match_index = np.argmin(face_distances)

    # если наименьшее значение расстояния, что вышло из результата меньше, чем 0.6, то скорее всего это лицо нам знакомо
    # грницу в 0.6 можно поставить другую, просто этот алгоритм обычно дает дистанцию меньше чем 0.6 для одного лица. Ещё берем запас в 0.5
    if face_distances[best_match_index] < 0.65:
        # берем данные по этому лицу
        metadata = known_face_metadata[best_match_index]

        # обновляем время последнего раза когда мы видим этого человека.
        metadata["last_seen"] = datetime.now()
        metadata["seen_frames"] += 1

        # если человека видели менее пяти минут назад, то это тот-же визит, если дольше то засчитывается как другое посещение
        if datetime.now() - metadata["first_seen_this_interaction"] > timedelta(minutes=5):
            metadata["first_seen_this_interaction"] = datetime.now()
            metadata["seen_count"] += 1

    return metadata

# функция для получения энкодинга с фото лица
def RetrieveFacesAndData(image, registerNewFaces = False):
    # уменьшим размер картинки в два раза для скорости работы
    small_frame = cv.resize(image, (0, 0), fx=0.5, fy=0.5)

    # меняем BGR, стандартный для OpenCV в RGB нужный для face_recognition
    rgb_small_frame = cv.cvtColor(small_frame, cv.COLOR_BGR2RGB)


    # находим места всех лиц и делаем их энкодинги в картинке
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    # в цикле проверяем не видели ли мы это лицо до этого и делаем подпись для рамки
    face_labels = []
    for face_location, face_encoding in zip(face_locations, face_encodings):
        # проверяем на то, что это лицо в списке известных лиц.
        metadata = lookup_known_face(face_encoding)

        # если лицо найдено, до делаем подпись
        if metadata is not None:
            HumanName = metadata['Name']
            time_at_door = datetime.now() - metadata['first_seen_this_interaction']
            face_label = f"{HumanName} is waiting {int(time_at_door.total_seconds())} secunds"

        # Если лицо полностью новое, то добавляем его в список
        else:
            face_label = "Unknown human"

            # вырезаем лицо с картинки
            top, right, bottom, left = face_location
            face_image = small_frame[top:bottom, left:right]
            face_image = cv.resize(face_image, (150, 150))
            # если мы указали, что новые лица надо сохранить, то сделаем это сразу
            if registerNewFaces == True:
                register_new_face(face_encoding, face_image)

        face_labels.append(face_label)
    return face_locations, face_encodings, face_labels




def main_loop():
    # получаем видеопоток с учетом платформы
    if platform.machine() == "aarch64":# если платформа jetson_nano
        # захват видео с камеры с добавлением необходимой подписи, нужной для jetson_nano
        video_capture = cv.VideoCapture(get_jetson_gstreamer_source(), cv.CAP_GSTREAMER)
    else:
        # стандартный вариант для openCV захват видео с камеры
        video_capture = cv.VideoCapture(0)
        # video_capture = cv.VideoCapture("video.mp4") # мы можем  передать конкретное видео для обработки, если нужно

    # переменная хранит информацию о том, сколько было увидено новых лиц в процессе этого запуска.
    # По достижении порога лица сохраняются в файл
    number_of_faces_since_save = 0

    while True:
        # берем один кадр из видео
        ret, frame = video_capture.read()
        face_locations, face_encodings, face_labels = RetrieveFacesAndData(frame, registerNewFaces = True)

        # рисуем квадратик для каждого лица и даем подпись
        for (top, right, bottom, left), face_label in zip(face_locations, face_labels):
            # умножаем каждую точку на два потому, что они были посчитаны для изображения уменьшеного в два раза
            top *= 2
            right *= 2
            bottom *= 2
            left *= 2

            # непосредственно квадрат
            cv.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # подпись под квадратиком
            cv.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv.FILLED)
            cv.putText(frame, face_label, (left + 6, bottom - 6), cv.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

        # показываем схраненную картинку человека
        number_of_recent_visitors = 0
        for metadata in known_face_metadata:
            # если в течение этой минуты мы видели этого человека
            if datetime.now() - metadata["last_seen"] < timedelta(seconds=10) and metadata["seen_frames"] > 5:
                # отобразим фотографию его лица, которую мы сохранили в метаданных
                x_position = number_of_recent_visitors * 150

                frame[30:180, x_position:x_position + 150] =  cv.resize(metadata["face_image"], (150,150), interpolation = cv.INTER_AREA)
                number_of_recent_visitors += 1

                # подписываем картинку именем и количеством посещений
                visit_label = f"{metadata['Name']} made {metadata['seen_count']} visits"
                # if visits == 1:
                #     visit_label = "первый раз"

                # добавляем текст
                cv.putText(frame, visit_label, (x_position + 10, 170), cv.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        # пишем сколько человек у двери
        if number_of_recent_visitors > 0:
            cv.putText(frame, "Visitors at Door", (5, 18), cv.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)



        # отображаем обработанный кадр видео
        cv.imshow('Video', frame)

        # для выключения программы надо нажать  'q'
        if cv.waitKey(1) & 0xFF == ord('q'):
            save_known_faces()
            break

        # сохраняем новые лица
        if len(face_locations) > 0 and number_of_faces_since_save > 100:
            save_known_faces()
            number_of_faces_since_save = 0
        else:
            number_of_faces_since_save += 1

    # если мы вышли из цикла, то нужно закрыть соединение с камерой и закрыть все окна, что мы создали
    video_capture.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    load_known_faces()
    main_loop()

