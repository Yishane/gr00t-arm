// SG90 arm firmware: 5 servos (base, shoulder, elbow, wrist, gripper).
//
// For the Arduino Nano (uses the standard Servo library).
//
// Serial protocol, 115200 baud, one command per line:
//   <a0>,<a1>,<a2>,<a3>,<a4>   target servo angles in degrees (floats ok)
//   ?                          reply with current angles "P a0,a1,a2,a3,a4"
//   S <deg_per_s>              set max slew speed (default 180)
// Servos move toward their targets at most SLEW deg/s so jumps in the
// command stream don't slam the arm. Targets are clamped to [MIN, MAX].
//
// Wiring: servo signal -> pins below. Servo red/brown -> external 5-6 V
// supply, NOT the board's 5V pin. Supply ground -> board GND.

#include <Servo.h>

const int PINS[5] = {3, 5, 6, 9, 10};  // base, shoulder, elbow, wrist, gripper

const int N = 5;
// Per-joint limits in servo degrees. Tighten these once the arm is built
// (software/calibrate.py helps find them) so it can't hit itself.
const float MIN_DEG[N] = {0, 0, 0, 0, 0};
const float MAX_DEG[N] = {180, 180, 180, 180, 180};
const float HOME[N] = {90, 90, 90, 90, 90};

// SG90 pulse range. 500-2400 us covers ~180 deg on most units.
const int US_MIN = 500;
const int US_MAX = 2400;

Servo servos[N];
float target[N];
float current[N];
float slew = 180.0;  // deg/s
unsigned long lastStep = 0;
char line[64];
int lineLen = 0;

float clampJoint(int i, float v) {
  if (v < MIN_DEG[i]) return MIN_DEG[i];
  if (v > MAX_DEG[i]) return MAX_DEG[i];
  return v;
}

void writeServo(int i, float deg) {
  int us = US_MIN + (int)((US_MAX - US_MIN) * deg / 180.0 + 0.5);
  servos[i].writeMicroseconds(us);
}

void handleLine(char *s) {
  if (s[0] == '?') {
    Serial.print("P ");
    for (int i = 0; i < N; i++) {
      Serial.print(current[i], 1);
      Serial.print(i < N - 1 ? ',' : '\n');
    }
    return;
  }
  if (s[0] == 'S') {
    float v = atof(s + 1);
    if (v > 0) slew = v;
    return;
  }
  float vals[N];
  int n = 0;
  char *tok = strtok(s, ",");
  while (tok && n < N) {
    vals[n++] = atof(tok);
    tok = strtok(NULL, ",");
  }
  if (n != N) return;  // ignore malformed lines
  for (int i = 0; i < N; i++) target[i] = clampJoint(i, vals[i]);
}

void setup() {
  Serial.begin(115200);
  for (int i = 0; i < N; i++) {
    servos[i].attach(PINS[i], US_MIN, US_MAX);
    target[i] = current[i] = HOME[i];
    writeServo(i, current[i]);
  }
  lastStep = millis();
  Serial.println("READY");
}

void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (lineLen > 0) {
        line[lineLen] = '\0';
        handleLine(line);
        lineLen = 0;
      }
    } else if (lineLen < (int)sizeof(line) - 1) {
      line[lineLen++] = c;
    }
  }

  unsigned long now = millis();
  if (now - lastStep >= 10) {
    float maxStep = slew * (now - lastStep) / 1000.0;
    lastStep = now;
    for (int i = 0; i < N; i++) {
      float d = target[i] - current[i];
      if (d > maxStep) d = maxStep;
      if (d < -maxStep) d = -maxStep;
      if (d != 0) {
        current[i] += d;
        writeServo(i, current[i]);
      }
    }
  }
}
