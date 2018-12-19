# Übung 1: Excercise ROS + Supervised Learning

									Jonas Heinke
									09.11.2018

*******************************************************************************
## Teil 1
-------------------------------------------------------------------------------
### 1.1) Make yourself familiar with all the existing code. But especially with 'catkin_ws/src/camera_pseudo/src/CameraPseudo.py' and 'ai_train/mnist_cnn_modified.py'
<-- $ rosrun camera_pseudo CameraPseudo.py -->
Programm CameraPseudo.py: Es handelt sich um ein lauffähiges Python-Programm inklusiv einer Main und einer Klasse CameraPseudo. Der Konstruktor der Klasse definiert mehrere Publishers (Talker) und einen Subscriber (Listener).
Die Publisherbezeichnungen, zum Beispiel: "self.publisher_specific_comprs", werden zum Aufruf innerhalb des Quellprogramms benötigt. Wichtige Parameter sind der Name des Topics, der Datentyp der Nachricht und gegebenfalls die Anzahl der Queue-Elemente. Weitere Parameter können vereinbart werden. Diese können auf der Seite "http://docs.ros.org/melodic/api/rospy/html/rospy.topics.Publisher-class.html" nachgelesen werden.
Ein Subscriber zum Empfang von Daten ist im Konstruktor vereinbart. Die wichtigsten Parameter sind der Topic (von dort werden die Daten abgeholt), der Datentyp der zu empfangenden Daten und ganz wichtig, die Bezeichnung der aufzurufenden Callbackmethode (http://docs.ros.org/melodic/api/rospy/html/rospy.topics.Subscriber-class.html). Die Callbackmethode wird im Falle eines Datenstroms aufgerufen und abgearbeitet. Ihr wird standardmäßig der Empfangsparameter übergeben. Sofern der Parameter "callback_args" parametriert worden ist, empfängt die Callbackmethode diesen als zweiten Wert.
Ferner werden noch Bilddaten "mnist" aus dem Internet geladen und temporär abgelegt, sozusagen zum Testen des Programms. In der Erweiterung können die zu analysierenden Bilddaten auch von einer WEB-Cam geliefert werden.
Das Main initialisiert den Node und erzeugt eine Instanz (Objekt) der Klasse, mit der dann wiederum die Klassenmethode "publish_data" gerufen wird.
Die Methode publish_data ruft ihrerseits weitere Methoden. Es wird aber zuvor geprüft ob ROS überhaut gestartet wurde. Auch die Verwendung der Kamera wird geprüft.
Es soll lediglich noch die Methode "publish spezifik" erörtert werden. Diese spielt in dieser Aufgabe noch eine Rolle. Hier wird ein, bereits im Konstruktor vereinbarter Publisher aufgerufen, um ein Bild "compressed imgmsg" an seinen Topic "/camera/output/specific/compressed_img_msgs" zu senden. Das Bild wird über einen Index [SPECIFIC_VALUE] dem Bildarchiv "mnist" entnommen.
Bei dem Python-Programm "ai_train/mnist_cnn_modified.py" handelt es sich um einen Trainingsalgorithmus "Convolutional Neural Network for MNIST". Genutzt wird die Bibliothek "keras" in Verbindung mit "tensorflow". Tensorflow, aber auch keras sind zu installieren. Zu den einzelnen Phasen der Verarbeitung wird in dem Abschnitt 1.3 etwas gesagt.
----------------------------------------------------------------------------------------
### 1.2) Write a subscriber to receive image data from the following topic "/camera/output/specific/compressed_img_msgs"
<-- $ rosrun camera_pseudo SubscibePicture.py -->
Das Pythonmodul "SubscibePicture.py" besitzt im Wesentlichen eine Main und eine Klasse. Die Main initialisiert den Node, erzeugt eine Instanz der Klasse und ruft eine Klassenmethode mit dem Subscriber auf. Über den Topic erhält der Subscriber ein Bild von einem Publisher des Moduls "CameraPseudo". Der zugehörige Callback "imageCallback" empfängt das Bild und erzeugt eine Log-Information (loginfo). Ferner besteht die Möglichkeit das empfangene Bild im JPG-Format abzuspeichern [../SOLUTION/B6L4_21:19:29.jpg]. Der zugehörige Graph befindet sich in der Datei ["/SOLUTION/CameraPseudo-SubscribePicture.png"].
--------------------------------------------------------------------------------
### 1.3 Train a model of your choice for the mnist data with keras (or use the predefined in `ai_train/models/`)
### Include your trained model in `catkin_ws/src/prediction/src/Prediction.py` and predict the value based on the subscribed image inside the subscribers callback (pay attention, that predicting in a callback is a different thread: https://stackoverflow.com/questions/46725323/keras-tensorflow-exception-while-predicting-from-multiple-threads
<-- $ rosrun prediction Prediction.py (import CnnModels.py) -->
Das Programm "Prediction.py" wurde erweitert. Zwei unterschiedliche Trainingsmodelle wurden in einem separaten Modul "CnnModels.py" in separaten Klassenstrukturen untergebracht. Eine Klasse dieses Moduls entspricht dem vorgegebenen Programmcode "MnistCnn" . Der Coder der anderen Klasse "MnistScnn" entstammt im Wesentlichen der Quelle "deep learning with python". In dem Programm "Prediction.py" wird eine Instanz von einer der gewünschten Klasse gebildet. Getestet wurden beide Varianten. Die Programmstuktur beider Klassen ist ähnlich. Im Konstruktor wird geprüft, ob ein Training bereits ausgefürht wurde. Wenn noch keine trainierten Daten geladen werden können "self.loadModel()", dann wird ein Training "self.modified()" ausgeführt. Das Trainingsergebnis wird in den Dateien "cnnModel.json/scnnModel, cnnModel.h5/scnnModel.h5" abgespeichert, die in Folge genutzt werden. Der Programmcode zum Speichern und Laden der Modelle lehnt sich an den Code der Internetseite "https://machinelearningmastery.com/save-load-keras-deep-learning-models/" an.
Insgesamt wurden die Klassen so strukturiert, dass die Teilfunktion in möglicht unabhängige Methoden gelegt wurden. Dadurch ist das Programm besser wartbar und auch übersichtlicher.
Es existieren die Methoden "predictionTestImage(self,index=6) und eine Methode "predictionImage(self, input_data)". Es handelt sich um Vorhersagemethoden. Über den Index wird ein einzelnes Bild aus der Testmenge angesprochen. Mit "input_data" wir ein einzelnen Bild an die Vorhersagemethode übergeben.

Im wesentlich werden folgende Phasen durchlaufen:
- Laden der Daten (Bilder mit handschriftlichen Ziffern von 0 bis 9)
- Definition des Modells
- Compilieren des Modells
- Fit (trainieren) des Modells
- Evaluieren des Modells
Am Ende jeder Methode wird zusätzlich ein Einzelbild einer Klasse zugeordnet. Da handschriftliche Ziffern bewertet werden, gibt es die Zuornungsklassen von 0 bis 9.
- Predict Picture-Content
Die Methoden geben den Vorhersagewert (Predict) und den wahren Wert (real-Label) an die aufrufende Methode zurück.
-> ACHTUNG: Der Parameter "activation='softmax'" funktioniert erst ab der Tensorflow-Version 1.4
-----------------------------------------------------------------------------------
### 1.4) Publish your predicted number to (real class number not the one-hot encoded one) "/camera/input/specific/number"
<-- rosrun prediction Prediction.py (self.publisherPredictionNumber=rospy.Publisher("/camera/input/specific/number", ...) -->
Die Klasse prediction enthält einen Publisher, der die Vorhersage (prediction) an den Topic "/camera/input/specific/number" sendet. Die Definition dieses Publishers befindet sich im Konstruktor der Klasse.
Das Programm CameraPseudo.py übernimmt den Vorhersagewert aus dem Topic mit dem Subscriber "rospy.Subscriber('/camera/input/specific/number',...). In einer der zugehörigen Callback-Methode "camera_specific callback" wird geprüft, ob diese Vorhersge richtig oder falsch ist (True/False). Das Ergebis dieses Vergleichs wird wiederum in einem Topic veröffentlicht (published) '/camera/output/specific/check' und steht dort zur Abholung bereit (siehe nächster Abschnitt 1.5).
### 1.5)  Subscribe to the following topic to verify your prediction  /camera/output/specific/check
<-- rosrun prediction Prediction.py (self.subscribeVerifyPrediction=rospy.Subscriber(name='/camera/output/specific/check', ...) -->
Der Subscriber des Moduls "Prediction.py" holt das Vergleichsergebnis aus dem Topic '/camera/output/specific/check' ab. Die zugehörigen Callbackmethode "callbackVerifyPrediction" gibt das Ergebnis auf dem Display aus.
Die Graphik [/home/jon/catkin_ws/SOLUTION/CameraPseudo=Prediction.png] stellt die bidirektionale Kommunikation zwischen den beiden Nods (Abschnitte 1.4 und Abschnit 1.5) dar.
********************************************************************************
## Teil 2
--------------------------------------------------------------------------------
## 2) Write a subscriber to receive image data from the following topic "/camera/output/random/compressed_img_msgs" and Subscribe for verification to "/camera/output/random/number" (Since the images do NOT stay consistent, the image and according value will be published simultaneously. Safe if locally for verification.)
<-- rosrun camera_pseudo SubscibeImages.py -->
Ein zufällig ausgewähltes Bild und das zugehörige Label werden mittels zweier Publisher in dem Modul "CameraPseudo.py", in der Methode "publish random(self, verbose=0)" veröffentlicht. Entsprechend der Aufgabenstellung werden diese Informationen in zwei unterschiedlichen Topics abgelegt. Um die Bild-Label-Zuordnung auch beim Lesen zu erhalten, müssen beide Teilinformationen quasigleichzeitg aus den Topics abgeholt werden. Das ist so gelöst, dass zuerst das Bild subscribed wird. Die zugehörige Calbeckmedode enthält einen zweiten Subscriber für das zugehörigen Label. Das bereits vorhandene Bild wird mit Hilfe des Parameters "callback_ args" einfach durchgereicht. Das Bild mit dem zugehörigen Label steht dann in der zweiten Callbackmethode zur Weiterverarbeitung zur Verügung. In diesem Fall werden die Daten ausgegeben. Es sei noch darauf hingewisen, dass mit dem Setzen des zusätzlichen Parameters "callback_args" die aufgerufene Callbackmethode automatisch zwei Übernahmeparameter erwartet. In diesem Fall als ersten Parameter das Label und als zweiten Parameter das Bild. Siehe "http://docs.ros.org/melodic/api/rospy/html/rospy.topics.Subscriber-class.html".
Die Grafik ist auch hier wieder als Bild der Anlage zu finden [CameraPseudo--SubscribeImages.png]. Bemerkung: In der Grafikoberfläche rqt_graph ist zur Sichtbarkeit beider Topics der Parameter "Nodes/Topics (aktiv)" auszuwählen.
********************************************************************************
## Teil 3: self, SOLUTION.md, /home/jon/catkin_ws/SOLUTION/CameraPseudo-SubscribePicture.png, /home/jon/catkin_ws/SOLUTION/CameraPseudo=Prediction.png, /home/jon/catkin_ws/SOLUTION/CameraPseudo- -SubscribeImages.png, ....
********************************************************************************
## Teil 4: Optional
--------------------------------------------------------------------------------
### 4.1 Make use of ROS network capability and use publishing camera data on one device and predicting on another (see ROS network tutorial therefore: "http://wiki.ros.org/ROS/Tutorials/MultipleMachines", do not use plain python. Use ROS on both machines/VMs)
Die Umsetzun dieser Aufgabe erfordert eine entsprechende Versuchsumgebung bestehend aus zwei Computern mit dem Betriebssysystem Ubuntu. Auf beiden Systemen wurden ROS und weitere notwendige Bibliotheken installiert.
- 		Computer 1 	         Computer 2
- Name  : 	space		         itubuntu
- User  : 	heinke		         itjonas
- PW    : 	jon		         itjonas
- IP    : 	192.168.178.38	         192.168178.36
- Master:   	$ roscore	         -----------

Beide Computer befinden sich in einem Netz, verbunden mittels einer Fritz.box.
Die Erreichbarkeit lässt sich mit $ping computername zum Beispiel mit $ping itubuntu prüfen.
Die Antwort ist entsprechend: 64 bytes from itubuntu.fritz.box (192.168.178.36): icmp_seq=8 ttl=64 time=0.167 ms
Nur ein Computer darf als Master definiert werden. In diesem Fall ist es der Rechner "space".
ROS wird so konfiguriert, dass von allen beteiligten Computern der gestartete Master verwendet werden kann:
"$ export ROS_MASTER_URI= http://space:11311". Danach können in beliebiger Reihenfolge die Programmodule auf den Computern mit rosrun gestartet werden.
-------------------------------------------------------------------------------
### 4.2) Adapt your application to a more complex problem
Zunächst wurde die Netzwerkfähigkeit mit den bereits bekannten Daten und Modellen geprüft.
Ziel ist die Kommunikation zwischen zwei Programmen, die auf zwei unterschiedlichen PC’s laufen. Das Programm "CameraPseudo.py" des Computers "space" sendet ein Bildes an das Programm "Prediction.py", welches auf dem Computer "itubuntu" installiert ist. Dort wird das zugehörige Bild bewertet. Zuvor läuft ein Trainingsprozess auf diesem Computer ab. Das Prognose-Ergebnis true oder false wird zurückgeschickt an das Programm CamerPseudo.py und dort ausgegeben.
__TERMINAL-AUSZÜGE (für den Test wurde die Anzahl der Epochen reduziert)__
* Master sendet ein Bild an den Client:
<--
jon@space:~/catkin_ws$ rosrun camera_pseudo CameraPseudo.py
Using TensorFlow backend.
Downloading data from https://s3.amazonaws.com/img-datasets/mnist.npz
11493376/11490434 [==============================] - 4s 0us/step
11501568/11490434 [==============================] - 4s 0us/step
[INFO] [1541578353.662842]: Publishing data… -->

* Client führt Training durch und bewertet das empfangene Bild. Schickt im Anschluss das Ergebnis (true/false) an den Client zurück:
<--
itjonas@itubuntu:~/catkin_ws$ rosrun prediction Prediction.py
Using TensorFlow backend.
try in main
Hier ist der Konstruktor
class CnnModels - Konstruktor
10000
Train on 60000 samples, validate on 10000 samples
Epoch 1/3
60000/60000 [==============================] - 1357s 23ms/step - loss: 0.1928 - acc: 0.9444 - val_loss: 0.0722 - val_acc: 0.9776
Epoch 2/3 ... usw. ... -->
...............................................................................
Die Grafik ist entsprechend: "/home/jon/catkin_ws/SOLUTION/CameraPseudo-Netz-Prediction.png". Leider ist keine Einstellung bekannt, die auch den Computer benennen, auf dem das jeweilige Modul läuft.
--------------------------------------------------------------------------------
### 4.3) Real images based on webcam input (you can find an implemented publisher for the webcam in `CameraPseudo.py`)
Hier wurde die Übertragung der WebCam-Bilder von einem Rechner zum anderen Rechner getestet.
Die Webcam des Rechners "itubuntu" sendet forlaufend Bilder an dern rechner "space". Auf dem rechner "space" werden diese auf dem Display vortlaufend dargestellt (itubutu ist in diesem Beispiel der Master).
<--
itjonas@itubuntu:~/catkin_ws$ rosrun camera_pseudo PublishWebCam.py
Using TensorFlow backend.
[INFO] [1542179332.658458]: Publish data... -->
<--
jon@space:~/catkin_ws$ rosrun camera_pseudo SubscriberCam.py
Using TensorFlow backend.
Subscriber Cam
subscribed from: /camera/output/webcam/compressed_img_msgs -->

Der PublisherWebCam.py (analog zu PseudoCamera.py) sendet die Bildfolge und der subscriberCam.py empfängt fortlaufen. Es können etwa 3 Bilder pro Sekunde übertragen werden.
Ein Screenshot des übertragenen Bildes: Bildschirmfoto PublishWebCam-SubscriberCam.png

### 4.4)  Adapt your application to a more complex problem, like cifar or real images based on webcam input (you can find an implemented publisher for the webcam in `CameraPseudo.py`)
Das Modul "CnnModels.py" besitzt in Erweiterung die Klasse "Cifar10scnn". Der Aufbau dieser Klasse ähnelt den vorangegangenen. Da es sich um Farbbilder mit der Größe 32x32 Pixel handelt, ist das Laden und die Datenaufbereitung der Traings und der Testdatenen angepasst.
Das verwendete Modell unterscheidet sich ebenfalls von den vorangegangenen Versionen.
In der Main des Moduls Prediction.py ist natürlich die gewünschte Klasse zu initialisieren. In diesem Fall: "cnn=CnnModels.Cifar10scnn()".
Das Speichern und das Laden des einmal trainierten Models geschieht genauso wie in den anderen Klassen. Eine zusätzlich Hilfsklasse, die gemeinsam verwendete Methoden enthält könnte weitere Vereinfachungen ermöglichen.
###############################################################################
