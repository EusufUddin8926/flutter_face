import 'package:flutter/material.dart';
import 'package:google_ml_kit/google_ml_kit.dart';

class FaceDetectorPainter extends CustomPainter {
  FaceDetectorPainter(this.imageSize, this.results);

  final Size imageSize;
  dynamic results;
  late double scaleX, scaleY;

  @override
  void paint(Canvas canvas, Size size) {
    final Paint paint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3.0
      ..color = Colors.greenAccent;

    // Calculate scaling factors
    scaleX = size.width / imageSize.width;
    scaleY = size.height / imageSize.height;

    // Iterate over detected faces
    for (String label in results.keys) {
      for (Face face in results[label]) {
        // Draw the bounding box for each face
        final scaledRect = _scaleRect(
          rect: face.boundingBox,
          scaleX: scaleX,
          scaleY: scaleY,
        );

        canvas.drawRRect(
          RRect.fromRectAndRadius(scaledRect, Radius.circular(10)),
          paint,
        );

        // Draw label near the top-left corner of the bounding box
        TextSpan span = TextSpan(
          style: TextStyle(color: Colors.orange[300], fontSize: 15),
          text: label,
        );
        TextPainter textPainter = TextPainter(
          text: span,
          textAlign: TextAlign.left,
          textDirection: TextDirection.ltr,
        );
        textPainter.layout();

        // Position the label
        textPainter.paint(
          canvas,
          Offset(scaledRect.left, scaledRect.top - 10), // Adjusting label position
        );
      }
    }
  }

  @override
  bool shouldRepaint(FaceDetectorPainter oldDelegate) {
    return oldDelegate.imageSize != imageSize || oldDelegate.results != results;
  }
}

// Scales the rectangle to the canvas size
Rect _scaleRect({
  required Rect rect,
  required double scaleX,
  required double scaleY,
}) {
  return Rect.fromLTRB(
    rect.left * scaleX,
    rect.top * scaleY,
    rect.right * scaleX,
    rect.bottom * scaleY,
  );
}
