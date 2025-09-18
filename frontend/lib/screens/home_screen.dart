// ignore_for_file: avoid_print

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'dart:io';
import 'package:http/http.dart' as http;
import 'dart:convert';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  File? _selectedImage;
  // ignore: non_constant_identifier_names
  Map<String, dynamic>? model_response;

  // ignore: non_constant_identifier_names
  Future<void> _pickImageFromGallery() async {
    final picker = await ImagePicker().pickImage(source: ImageSource.gallery);
    if (picker != null) {
      setState(() {
        _selectedImage = File(picker.path);
        // Print the selected image path
        print('Selected image path: ${picker.path}');
      });

      // Optionally send to backend immediately after picking
      // await _sendImageToBackend(_selectedImage!, 'your_model_name');
    }
  }

  Future<void> _sendImageToBackend(File image, String modelName) async {
    setState(() {
      model_response = null;
    });

    // Print the path being sent
    print('Sending image path to backend: ${image.path}');
    print('Sending model name: $modelName');

    try {
      // Prepare the request body
      Map<String, dynamic> requestBody = {
        'image_path': image.path,
        'model_name': modelName,
      };

      // Print request details before sending
      print('Request prepared with:');
      print('- Model: $modelName');
      print('- Image path: ${image.path}');

      // Send POST request
      var response = await http.post(
        Uri.parse('http://127.0.0.1:8000/predict'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode(requestBody),
      );

      // Handle response
      if (response.statusCode == 200) {
        print('Backend response: ${response.body}');
        setState(() {
          model_response = jsonDecode(response.body);
        });
      } else {
        print('Error from backend: ${response.statusCode} - ${response.body}');
      }
    } catch (e) {
      print('Exception while sending to backend: $e');
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SingleChildScrollView(
        child: Center(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              SizedBox(height: 30),
              Text(
                "AI-Powered Bone Marrow",
                style: TextStyle(fontWeight: FontWeight.bold, fontSize: 40),
              ),
              Text(
                'Cell Analysis',
                style: TextStyle(
                  fontWeight: FontWeight.bold,
                  fontSize: 40,
                  color: Colors.blue,
                ),
              ),
              Container(
                width: 500,
                height: 70,
                alignment: Alignment.center,
                child: Text(
                  'Advanced Diagnostic tool for classifying bone marrow cells and detecting abnomalities using state-of-the-art-models.',
                  textAlign: TextAlign.center,
                ),
              ),

              // upload image logic here
              ElevatedButton(
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.blue,
                  foregroundColor: Colors.white,
                  fixedSize: Size(170, 60),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(10),
                  ),
                ),
                onPressed: () {
                  _pickImageFromGallery();
                },
                child: Row(
                  children: [
                    Icon(Icons.upload),
                    SizedBox(width: 8),
                    Flexible(
                      child: Text(
                        'Upload Image to diagnose',
                        softWrap: true,
                        textAlign: TextAlign.center,
                      ),
                    ),
                  ],
                ),
              ),

              Padding(
                padding: const EdgeInsets.only(
                  left: 10,
                  right: 10,
                  top: 20,
                  bottom: 20,
                ),
                child: Container(
                  width: 500,
                  height: 300,
                  child: _selectedImage != null
                      ? Image.file(_selectedImage!, fit: BoxFit.cover)
                      : null,
                ),
              ),
              Text(
                "Advanced Daignostic Features",
                style: TextStyle(fontWeight: FontWeight.bold, fontSize: 25),
              ),
              Text(
                'Cutting edge technology for accurate bone marrow cell analysis',
              ),
              SizedBox(height: 20),
              // now the next page model selection page
              Container(
                width: 500,
                height: 300,
                child: Image.asset('assets/images/Allogenic-bone-1.jpeg'),
              ),
              // the model selection choices
              Padding(
                padding: const EdgeInsets.all(8),
                child: Container(
                  padding: EdgeInsets.all(20),
                  width: 850,
                  height: 460,
                  decoration: BoxDecoration(
                    color: Colors.grey[50],
                    shape: BoxShape.rectangle,
                    border: Border.all(color: Colors.purple),
                  ),
                  child: Column(
                    children: [
                      Text(
                        'Select AI Model',
                        style: TextStyle(
                          fontSize: 20,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      Text('Choose the most suitable model for your analysis'),
                      SizedBox(height: 50),
                      Row(
                        mainAxisAlignment: MainAxisAlignment.spaceAround,
                        children: [
                          Material(
                            color: Colors.transparent,
                            child: Container(
                              height: 120,
                              width: 240,
                              decoration: BoxDecoration(border: Border.all()),
                              child: InkWell(
                                onTap: () {
                                  if (_selectedImage != null) {
                                    _sendImageToBackend(
                                      _selectedImage!,
                                      'Inception',
                                    );
                                  }
                                },
                                hoverColor: Colors.purple[50],
                                child: Padding(
                                  padding: const EdgeInsets.all(8.0),
                                  child: Column(
                                    spacing: 20,
                                    children: [
                                      Row(
                                        mainAxisAlignment:
                                            MainAxisAlignment.spaceBetween,
                                        children: [
                                          Text(
                                            'Inception Model',
                                            style: TextStyle(
                                              fontWeight: FontWeight.bold,
                                            ),
                                          ),
                                          Icon(Icons.layers),
                                        ],
                                      ),
                                      Text(
                                        'Highly efficient for detecting complex patterns in images',
                                      ),
                                    ],
                                  ),
                                ),
                              ),
                            ),
                          ),
                          Material(
                            color: Colors.transparent,
                            child: Container(
                              height: 120,
                              width: 240,
                              decoration: BoxDecoration(border: Border.all()),
                              child: InkWell(
                                onTap: () {
                                  if (_selectedImage != null) {
                                    _sendImageToBackend(
                                      _selectedImage!,
                                      'ResNet',
                                    );
                                  }
                                },
                                hoverColor: Colors.purple[50],
                                child: Padding(
                                  padding: const EdgeInsets.all(8.0),
                                  child: Column(
                                    spacing: 20,
                                    children: [
                                      Row(
                                        mainAxisAlignment:
                                            MainAxisAlignment.spaceBetween,
                                        children: [
                                          Text(
                                            'ResNet (Residual Network)',
                                            style: TextStyle(
                                              fontWeight: FontWeight.bold,
                                            ),
                                          ),
                                          Icon(Icons.layers),
                                        ],
                                      ),
                                      Text(
                                        'Deep Learning model for high accuracy in classification',
                                      ),
                                    ],
                                  ),
                                ),
                              ),
                            ),
                          ),
                          Material(
                            color: Colors.transparent,
                            child: Container(
                              height: 120,
                              width: 240,
                              decoration: BoxDecoration(border: Border.all()),
                              child: InkWell(
                                onTap: () {
                                  if (_selectedImage != null) {
                                    _sendImageToBackend(
                                      _selectedImage!,
                                      'MobileNet',
                                    );
                                  }
                                },
                                hoverColor: Colors.purple[50],
                                child: Padding(
                                  padding: const EdgeInsets.all(8.0),
                                  child: Column(
                                    spacing: 20,
                                    children: [
                                      Row(
                                        mainAxisAlignment:
                                            MainAxisAlignment.spaceBetween,
                                        children: [
                                          Text(
                                            'MobileNet',
                                            style: TextStyle(
                                              fontWeight: FontWeight.bold,
                                            ),
                                          ),
                                          Icon(Icons.layers),
                                        ],
                                      ),
                                      Text(
                                        'Neural Network used for less resources',
                                      ),
                                    ],
                                  ),
                                ),
                              ),
                            ),
                          ),
                        ],
                      ),
                    ],
                  ),
                ),
              ),

              // the last page results.
              SizedBox(height: 100),
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Container(
                    padding: EdgeInsets.all(8),
                    width: 350,
                    height: 500,
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'Analysis Result',
                          style: TextStyle(
                            fontSize: 15,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                        SizedBox(height: 25),
                        Text('Cell Type detected'),
                        SizedBox(height: 15),
                        Divider(color: Colors.black),
                        Visibility(
                          visible:
                              model_response != null &&
                              model_response?['cell_type'] != null &&
                              model_response?['confidence'] != null,
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              // Cell Type Display
                              Text(
                                model_response?['cell_type']?.toString() ??
                                    'Unknown cell type',
                                style: const TextStyle(
                                  fontSize: 16,
                                  fontWeight: FontWeight.bold,
                                ),
                              ),
                              const Divider(color: Colors.black),
                              const SizedBox(height: 15),

                              // Confidence Display
                              const Text('Model Confidence'),
                              const SizedBox(height: 8),
                              LinearProgressIndicator(
                                value:
                                    (model_response?['confidence'] as num?)
                                        ?.toDouble() ??
                                    0.0,
                                valueColor: const AlwaysStoppedAnimation(
                                  Colors.blue,
                                ),
                                minHeight: 9,
                              ),
                              const SizedBox(height: 4),
                              Text(
                                '${(((model_response?['confidence'] as num?)?.toDouble() ?? 0.0) * 100).toStringAsFixed(1)}%',
                                style: const TextStyle(fontSize: 12),
                              ),
                            ],
                          ),
                        ),
                      ],
                    ),
                  ),
                  Image.asset(
                    'assets/images/cells-image.png',
                    width: 300,
                    height: 400,
                    fit: BoxFit.cover,
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }
}
