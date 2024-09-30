import 'dart:convert';
import 'dart:io';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:google_ml_kit/google_ml_kit.dart';
import 'package:image/image.dart' as imglib;
import 'package:path_provider/path_provider.dart';
import 'package:quiver/collection.dart';
import 'package:tflite_flutter/tflite_flutter.dart' as tfl;
import 'package:tflite_flutter/tflite_flutter.dart';
import 'detector_painters.dart';
import 'utils.dart';

void main() {
  runApp(MaterialApp(
    themeMode: ThemeMode.light,
    theme: ThemeData(brightness: Brightness.light),
    home: _MyHomePage(),
    title: "Face Recognition",
    debugShowCheckedModeBanner: false,
  ));
}

class _MyHomePage extends StatefulWidget {
  @override
  _MyHomePageState createState() => _MyHomePageState();
}

class _MyHomePageState extends State<_MyHomePage> {
  late File jsonFile;
  dynamic _scanResults;
  CameraController? _camera;
  var interpreter;
  bool _isDetecting = false;
  CameraLensDirection _direction = CameraLensDirection.back;
  dynamic data = {};
  double threshold = 0.6;
  late Directory tempDir;
  List e1 = [];
  bool _faceFound = false;
  final TextEditingController _name = TextEditingController();

  @override
  void initState() {
    super.initState();
    SystemChrome.setPreferredOrientations([DeviceOrientation.portraitUp]);
    _initializeCamera();
  }

  Future<void> loadModel() async {
    late Delegate delegate;
    try {
      if (Platform.isAndroid) {
        delegate = tfl.GpuDelegateV2(
          options: tfl.GpuDelegateOptionsV2(
            isPrecisionLossAllowed: false,
            inferencePriority1: tfl.TfLiteGpuInferencePriority.minLatency,
            // Priority 1
            inferencePriority2: tfl.TfLiteGpuInferencePriority.auto,
            // Priority 2
            inferencePriority3: tfl.TfLiteGpuInferencePriority.auto,
          ),
        );
      }else{
        delegate = GpuDelegate(
          options: GpuDelegateOptions(
              allowPrecisionLoss: true,
              waitType: TFLGpuDelegateWaitType.active),
        );
      }

      var interpreterOptions = InterpreterOptions()..addDelegate(delegate);
      interpreter = await tfl.Interpreter.fromAsset('mobilefacenet.tflite',
          options: interpreterOptions);
    } catch (e) {
      print('Failed to load model: $e');
    }
  }

  void _initializeCamera() async {
    await loadModel();

    // Get camera description for the specified direction
    CameraDescription description = await getCamera(_direction);
    InputImageRotation rotation =
        rotationIntToImageRotation(description.sensorOrientation);

    // Initialize the camera
    _camera =
        CameraController(description, ResolutionPreset.low, enableAudio: false);
    await _camera!.initialize();
    await Future.delayed(Duration(milliseconds: 500));

    // Setup the temporary directory and JSON file for embeddings
    tempDir = await getApplicationDocumentsDirectory();
    String _embPath = '${tempDir.path}/emb.json';
    jsonFile = File(_embPath);
    if (jsonFile.existsSync()) {
      data = json.decode(jsonFile.readAsStringSync());
    }

    // Start image stream for face detection
    _camera!.startImageStream((CameraImage image) {
      if (_camera != null && !_isDetecting) {
        _isDetecting = true;
        dynamic finalResult = Multimap<String, Face>();

        // Convert the camera image to InputImage for ML Kit
        InputImage inputImage = _convertCameraImageToInputImage(image, _direction);

        // Detect faces using ML Kit's face detector
        detect(image, _getDetectionMethod(), rotation)
            .then((dynamic result) async {
          _faceFound = result.isNotEmpty;
          Face _face;

          // Convert the camera image to a format suitable for cropping
          imglib.Image convertedImage = _convertCameraImage(image, _direction);

          for (_face in result) {
            // Crop and resize the face from the image
            double x = _face.boundingBox.left - 10;
            double y = _face.boundingBox.top - 10;
            double w = _face.boundingBox.width + 10;
            double h = _face.boundingBox.height + 10;
            imglib.Image croppedImage = imglib.copyCrop(
                convertedImage, x.round(), y.round(), w.round(), h.round());
            croppedImage = imglib.copyResizeCropSquare(croppedImage, 112);

            // Recognize the face
            String res = _recog(croppedImage);
            finalResult.add(res, _face);
          }

          setState(() {
            _scanResults = finalResult;
          });

          _isDetecting = false;
        }).catchError((_) {
          _isDetecting = false;
        });
      }
    });
  }

  // Helper method to convert CameraImage to InputImage for Google ML Kit
  InputImage _convertCameraImageToInputImage(CameraImage image, CameraLensDirection direction) {
    if (Platform.isAndroid) {
      // Android specific: Handle NV21
      final WriteBuffer allBytes = WriteBuffer();
      for (Plane plane in image.planes) {
        allBytes.putUint8List(plane.bytes);
      }
      final bytes = allBytes.done().buffer.asUint8List();

      final Size imageSize = Size(image.width.toDouble(), image.height.toDouble());

      final InputImageRotation imageRotation =
          InputImageRotationMethods.fromRawValue(
              _camera!.description.sensorOrientation) ??
              InputImageRotation.Rotation_0deg;

      final InputImageFormat inputImageFormat =
          InputImageFormatMethods.fromRawValue(image.format.raw) ??
              InputImageFormat.NV21;

      final planeData = image.planes.map(
            (Plane plane) {
          return InputImagePlaneMetadata(
            bytesPerRow: plane.bytesPerRow,
            height: plane.height,
            width: plane.width,
          );
        },
      ).toList();

      return InputImage.fromBytes(
        bytes: bytes,
        inputImageData: InputImageData(
          size: imageSize,
          imageRotation: imageRotation,
          inputImageFormat: inputImageFormat,
          planeData: planeData,
        ),
      );
    } else if (Platform.isIOS) {
      return InputImage.fromBytes(
        bytes: image.planes[0].bytes, // Simplified for iOS
        inputImageData: InputImageData(
          size: Size(image.width.toDouble(), image.height.toDouble()),
          imageRotation: InputImageRotation.Rotation_0deg, // Adjust as needed
          inputImageFormat: InputImageFormat.YUV420, // Use YUV420 for iOS
          planeData: [], // iOS doesn't expose plane details like Android
        ),
      );
    } else {
      throw UnsupportedError("Platform not supported");
    }
  }

  HandleDetection _getDetectionMethod() {
    final FaceDetector faceDetector = GoogleMlKit.vision.faceDetector(
      const FaceDetectorOptions(
        mode: FaceDetectorMode.accurate,
        enableClassification: false,
      ),
    );

    return (InputImage inputImage) async {
      // Process the InputImage and return detected faces
      List<Face> faces = await faceDetector.processImage(inputImage);
      return faces; // Return the list of detected faces
    };
  }

  Widget _buildResults() {
    if (_scanResults == null ||
        _camera == null ||
        !_camera!.value.isInitialized) {
      return const Text('');
    }
    CustomPainter painter;

    final Size imageSize = Size(
        _camera!.value.previewSize!.height, _camera!.value.previewSize!.width);
    painter = FaceDetectorPainter(imageSize, _scanResults);
    return CustomPaint(
      painter: painter,
    );
  }

  Widget _buildImage() {
    if (_camera == null || !_camera!.value.isInitialized) {
      return const Center(child: CircularProgressIndicator());
    }

    return Container(
      constraints: const BoxConstraints.expand(),
      child: Stack(
        fit: StackFit.expand,
        children: <Widget>[
          CameraPreview(_camera!),
          _buildResults(),
        ],
      ),
    );
  }

  void _toggleCameraDirection() async {
    if (_direction == CameraLensDirection.back) {
      _direction = CameraLensDirection.front;
    } else {
      _direction = CameraLensDirection.back;
    }
    await _camera!.stopImageStream();
    await _camera!.dispose();

    setState(() {
      _camera = null;
    });

    _initializeCamera();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Face recognition'),
        actions: <Widget>[
          PopupMenuButton<Choice>(
            onSelected: (Choice result) {
              if (result == Choice.delete) {
                _resetFile();
              } else {
                _viewLabels();
              }
            },
            itemBuilder: (BuildContext context) => <PopupMenuEntry<Choice>>[
              const PopupMenuItem<Choice>(
                child: Text('View Saved Faces'),
                value: Choice.view,
              ),
              const PopupMenuItem<Choice>(
                child: Text('Remove all faces'),
                value: Choice.delete,
              ),
            ],
          ),
        ],
      ),
      body: _buildImage(),
      floatingActionButton: Column(
        mainAxisAlignment: MainAxisAlignment.end,
        children: [
          FloatingActionButton(
            backgroundColor: (_faceFound) ? Colors.blue : Colors.blueGrey,
            child: const Icon(Icons.add),
            onPressed: () {
              if (_faceFound) _addLabel();
            },
          ),
          const SizedBox(height: 10),
          FloatingActionButton(
            onPressed: _toggleCameraDirection,
            child: _direction == CameraLensDirection.back
                ? const Icon(Icons.camera_front)
                : const Icon(Icons.camera_rear),
          ),
        ],
      ),
    );
  }

  imglib.Image _convertCameraImage(
      CameraImage image, CameraLensDirection _dir) {
    int width = image.width;
    int height = image.height;
    var img = imglib.Image(width, height);
    const int hexFF = 0xFF000000;
    final int uvyButtonStride = image.planes[1].bytesPerRow;
    final int? uvPixelStride = image.planes[1].bytesPerPixel;
    for (int x = 0; x < width; x++) {
      for (int y = 0; y < height; y++) {
        final int uvIndex = uvPixelStride! * (x / 2).floor() +
            uvyButtonStride * (y / 2).floor();
        final int index = y * width + x;
        final yp = image.planes[0].bytes[index];
        final up = image.planes[1].bytes[uvIndex];
        final vp = image.planes[2].bytes[uvIndex];
        int r = (yp + vp * 1436 / 1024 - 179).round().clamp(0, 255);
        int g = (yp - up * 46549 / 131072 + 44 - vp * 93604 / 131072 + 91)
            .round()
            .clamp(0, 255);
        int b = (yp + up * 1814 / 1024 - 227).round().clamp(0, 255);
        img.data[index] = hexFF | (b << 16) | (g << 8) | r;
      }
    }
    return (_dir == CameraLensDirection.front)
        ? imglib.copyRotate(img, -90)
        : imglib.copyRotate(img, 90);
  }

  String _recog(imglib.Image img) {
    List input = imageToByteListFloat32(img, 112, 128, 128);
    input = input.reshape([1, 112, 112, 3]);
    List output = List.filled(1 * 192, 0).reshape([1, 192]);
    interpreter.run(input, output);
    output = output.reshape([192]);
    e1 = List.from(output);
    String result = compare(e1).toUpperCase();
    return result;
  }

  void _addLabel() {
    showDialog(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          title: const Text("Add Face"),
          content: TextField(
            controller: _name,
          ),
          actions: <Widget>[
            ElevatedButton(
              child: const Text("Cancel"),
              onPressed: () {
                Navigator.of(context).pop();
              },
            ),
            ElevatedButton(
              child: const Text("Save"),
              onPressed: () {
                String name = _name.text;
                _name.clear();
                data[name] = e1;
                jsonFile.writeAsStringSync(json.encode(data));
                Navigator.of(context).pop();
              },
            ),
          ],
        );
      },
    );
  }

  void _resetFile() {
    jsonFile.deleteSync();
    data = {};
  }

  String compare(List currEmb) {
    if (data.length == 0) return "No Face saved";
    double minDist = 999;
    double currDist = 0.0;
    String predRes = "NOT RECOGNIZED";
    for (String label in data.keys) {
      currDist = euclideanDistance(data[label], currEmb);
      if (currDist <= threshold && currDist < minDist) {
        minDist = currDist;
        predRes = label;
      }
    }
    print(minDist.toString() + " " + predRes);
    return predRes;
  }

  void _viewLabels() {
    showDialog(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          title: const Text("Saved Faces"),
          content: SizedBox(
            width: 300,
            height: 400,
            child: ListView.builder(
              shrinkWrap: true,
              itemCount: data.length,
              itemBuilder: (BuildContext context, int index) {
                return ListTile(
                  title: Text(data.keys.elementAt(index)),
                );
              },
            ),
          ),
          actions: <Widget>[
            ElevatedButton(
              child: const Text("OK"),
              onPressed: () {
                Navigator.of(context).pop();
              },
            ),
          ],
        );
      },
    );
  }
}

enum Choice { view, delete }
