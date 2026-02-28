/*
 * NERF Turret Servo Controller v3.0 - Arduino Firmware
 * =====================================================
 * Features:
 * - Watchdog timer: auto-center and disarm if no command in 2 seconds
 * - Heartbeat: sends HB:<pan>,<tilt>,<armed> every 500ms
 * - Acceleration-limited movement for smooth physical motion
 * - Extended command protocol: P/T/F, C (center), D (disarm), S (status), R (reboot)
 * - EEPROM-stored center offsets and limits
 *
 * Protocol (ASCII):
 *   Receive: "P<pan>T<tilt>F<fire>\n"   - Standard position/fire
 *            "C\n"                        - Center servos
 *            "D\n"                        - Disarm (stop firing, center)
 *            "S\n"                        - Status query
 *            "R\n"                        - Soft reboot
 *            "CFG:PO<val>TO<val>\n"       - Set center offsets
 *
 *   Respond: "OK\n"
 *            "ERR:<message>\n"
 *            "HB:<pan>,<tilt>,<armed>\n"  - Heartbeat
 *            "ST:<pan>,<tilt>,<armed>,<firing>,<smooth_pan>,<smooth_tilt>\n" - Full status
 *
 * Wiring:
 *   Pan servo  -> Pin 9  (PWM)
 *   Tilt servo -> Pin 10 (PWM)
 *   Fire relay -> Pin 7  (Digital)
 *   Status LED -> Pin 13 (built-in)
 */

#include <Servo.h>
#include <EEPROM.h>

// Pin definitions
#define PAN_SERVO_PIN   9
#define TILT_SERVO_PIN  10
#define FIRE_RELAY_PIN  7
#define STATUS_LED_PIN  13

// EEPROM addresses
#define EEPROM_MAGIC_ADDR   0
#define EEPROM_PAN_OFF_ADDR 2
#define EEPROM_TILT_OFF_ADDR 4
#define EEPROM_PAN_MIN_ADDR 6
#define EEPROM_PAN_MAX_ADDR 8
#define EEPROM_TILT_MIN_ADDR 10
#define EEPROM_TILT_MAX_ADDR 12
#define EEPROM_MAGIC_VALUE  0xAB

// Defaults (overridden by EEPROM if valid)
int panMin = 0;
int panMax = 180;
int tiltMin = 30;
int tiltMax = 150;
int panOffset = 90;
int tiltOffset = 90;

// Serial
#define BAUD_RATE 115200
#define CMD_BUFFER_SIZE 48

Servo panServo;
Servo tiltServo;

char cmdBuffer[CMD_BUFFER_SIZE];
int cmdIndex = 0;

int targetPan = 90;
int targetTilt = 90;
bool armed = false;
bool firing = false;
unsigned long fireStartTime = 0;
unsigned long fireDurationMs = 200;

// Acceleration-limited smooth movement
float smoothPan = 90.0;
float smoothTilt = 90.0;
float velPan = 0.0;
float velTilt = 0.0;
float maxAcceleration = 800.0;  // degrees/sec^2
float maxVelocity = 300.0;      // degrees/sec
unsigned long lastUpdateMicros = 0;

// Watchdog
unsigned long lastCommandTime = 0;
#define WATCHDOG_TIMEOUT_MS 2000

// Heartbeat
unsigned long lastHeartbeatTime = 0;
#define HEARTBEAT_INTERVAL_MS 500

// Status LED blink
unsigned long lastLedToggle = 0;
bool ledState = false;

void setup() {
    Serial.begin(BAUD_RATE);

    panServo.attach(PAN_SERVO_PIN);
    tiltServo.attach(TILT_SERVO_PIN);

    pinMode(FIRE_RELAY_PIN, OUTPUT);
    pinMode(STATUS_LED_PIN, OUTPUT);

    digitalWrite(FIRE_RELAY_PIN, LOW);

    loadEEPROMConfig();

    smoothPan = panOffset;
    smoothTilt = tiltOffset;
    targetPan = panOffset;
    targetTilt = tiltOffset;

    panServo.write(panOffset);
    tiltServo.write(tiltOffset);

    lastUpdateMicros = micros();
    lastCommandTime = millis();

    delay(500);
    Serial.println("NERF_TURRET_READY_V3");

    // Startup LED flash
    for (int i = 0; i < 3; i++) {
        digitalWrite(STATUS_LED_PIN, HIGH);
        delay(100);
        digitalWrite(STATUS_LED_PIN, LOW);
        delay(100);
    }
}

void loop() {
    unsigned long now = millis();

    // Read serial commands
    while (Serial.available() > 0) {
        char c = Serial.read();
        if (c == '\n' || c == '\r') {
            if (cmdIndex > 0) {
                cmdBuffer[cmdIndex] = '\0';
                processCommand(cmdBuffer);
                cmdIndex = 0;
                lastCommandTime = now;
            }
        } else if (cmdIndex < CMD_BUFFER_SIZE - 1) {
            cmdBuffer[cmdIndex++] = c;
        }
    }

    // Acceleration-limited smooth movement
    unsigned long nowMicros = micros();
    float dt = (nowMicros - lastUpdateMicros) / 1000000.0;
    lastUpdateMicros = nowMicros;
    dt = constrain(dt, 0.0001, 0.05);

    smoothPan = accelStep(smoothPan, targetPan, velPan, maxAcceleration, maxVelocity, dt);
    smoothTilt = accelStep(smoothTilt, targetTilt, velTilt, maxAcceleration, maxVelocity, dt);

    panServo.write(constrain((int)(smoothPan + 0.5), panMin, panMax));
    tiltServo.write(constrain((int)(smoothTilt + 0.5), tiltMin, tiltMax));

    // Handle fire duration
    if (firing && (now - fireStartTime >= fireDurationMs)) {
        firing = false;
        digitalWrite(FIRE_RELAY_PIN, LOW);
    }

    // Watchdog: if no command received in WATCHDOG_TIMEOUT_MS, disarm and center
    if (now - lastCommandTime > WATCHDOG_TIMEOUT_MS) {
        if (armed || firing) {
            armed = false;
            firing = false;
            digitalWrite(FIRE_RELAY_PIN, LOW);
            targetPan = panOffset;
            targetTilt = tiltOffset;
        }
    }

    // Heartbeat
    if (now - lastHeartbeatTime >= HEARTBEAT_INTERVAL_MS) {
        lastHeartbeatTime = now;
        sendHeartbeat();
    }

    // Status LED: solid when armed, blink when idle
    if (armed) {
        digitalWrite(STATUS_LED_PIN, HIGH);
    } else {
        if (now - lastLedToggle > 1000) {
            lastLedToggle = now;
            ledState = !ledState;
            digitalWrite(STATUS_LED_PIN, ledState);
        }
    }

    delay(5); // ~200Hz update rate
}

// ── Acceleration-limited step ──────────────────────────

float accelStep(float current, float target, float &velocity, float maxAcc, float maxVel, float dt) {
    float error = target - current;

    if (abs(error) < 0.2 && abs(velocity) < 1.0) {
        velocity = 0;
        return target;
    }

    float direction = (error > 0) ? 1.0 : -1.0;

    // Stopping distance at current velocity
    float stopDist = (velocity * velocity) / (2.0 * maxAcc);

    if (abs(error) <= stopDist + 0.5) {
        // Decelerate
        velocity -= direction * maxAcc * dt;
    } else {
        // Accelerate
        velocity += direction * maxAcc * dt;
    }

    velocity = constrain(velocity, -maxVel, maxVel);
    float newPos = current + velocity * dt;

    // Overshoot protection
    if ((error > 0 && newPos > target) || (error < 0 && newPos < target)) {
        newPos = target;
        velocity = 0;
    }

    return newPos;
}

// ── Command Processing ─────────────────────────────────

void processCommand(const char* cmd) {
    if (cmd[0] == 'C' && strlen(cmd) == 1) {
        targetPan = panOffset;
        targetTilt = tiltOffset;
        Serial.println("OK");
        return;
    }

    if (cmd[0] == 'D' && strlen(cmd) == 1) {
        armed = false;
        firing = false;
        digitalWrite(FIRE_RELAY_PIN, LOW);
        targetPan = panOffset;
        targetTilt = tiltOffset;
        Serial.println("OK");
        return;
    }

    if (cmd[0] == 'S' && strlen(cmd) == 1) {
        sendFullStatus();
        return;
    }

    if (cmd[0] == 'R' && strlen(cmd) == 1) {
        Serial.println("OK:REBOOTING");
        delay(100);
        asm volatile ("jmp 0");
        return;
    }

    if (strncmp(cmd, "ARM", 3) == 0) {
        armed = true;
        Serial.println("OK:ARMED");
        return;
    }

    // CFG:PO<val>TO<val> - configure center offsets
    if (strncmp(cmd, "CFG:", 4) == 0) {
        processConfigCommand(cmd + 4);
        return;
    }

    // Standard: P###T###F#
    if (strlen(cmd) < 9) {
        Serial.println("ERR:CMD_TOO_SHORT");
        return;
    }

    if (cmd[0] != 'P' || cmd[4] != 'T' || cmd[8] != 'F') {
        Serial.println("ERR:BAD_FORMAT");
        return;
    }

    char panStr[4] = {cmd[1], cmd[2], cmd[3], '\0'};
    int pan = atoi(panStr);

    char tiltStr[4] = {cmd[5], cmd[6], cmd[7], '\0'};
    int tilt = atoi(tiltStr);

    int fire = cmd[9] - '0';

    targetPan = constrain(pan, panMin, panMax);
    targetTilt = constrain(tilt, tiltMin, tiltMax);

    if (fire == 1 && armed && !firing) {
        firing = true;
        fireStartTime = millis();
        digitalWrite(FIRE_RELAY_PIN, HIGH);
    }

    Serial.println("OK");
}

void processConfigCommand(const char* cfg) {
    // Parse PO<val>TO<val>
    if (cfg[0] == 'P' && cfg[1] == 'O') {
        int po = atoi(cfg + 2);
        if (po >= 0 && po <= 180) {
            panOffset = po;
            EEPROM.put(EEPROM_PAN_OFF_ADDR, (int16_t)panOffset);
        }
    }

    const char* toPtr = strstr(cfg, "TO");
    if (toPtr) {
        int to = atoi(toPtr + 2);
        if (to >= 0 && to <= 180) {
            tiltOffset = to;
            EEPROM.put(EEPROM_TILT_OFF_ADDR, (int16_t)tiltOffset);
        }
    }

    // Write magic to confirm valid EEPROM
    EEPROM.put(EEPROM_MAGIC_ADDR, (int16_t)EEPROM_MAGIC_VALUE);

    Serial.print("OK:CFG_PO");
    Serial.print(panOffset);
    Serial.print("_TO");
    Serial.println(tiltOffset);
}

// ── Heartbeat & Status ─────────────────────────────────

void sendHeartbeat() {
    Serial.print("HB:");
    Serial.print((int)(smoothPan + 0.5));
    Serial.print(",");
    Serial.print((int)(smoothTilt + 0.5));
    Serial.print(",");
    Serial.println(armed ? "1" : "0");
}

void sendFullStatus() {
    Serial.print("ST:");
    Serial.print(targetPan);
    Serial.print(",");
    Serial.print(targetTilt);
    Serial.print(",");
    Serial.print(armed ? "1" : "0");
    Serial.print(",");
    Serial.print(firing ? "1" : "0");
    Serial.print(",");
    Serial.print((int)(smoothPan + 0.5));
    Serial.print(",");
    Serial.println((int)(smoothTilt + 0.5));
}

// ── EEPROM Config ──────────────────────────────────────

void loadEEPROMConfig() {
    int16_t magic;
    EEPROM.get(EEPROM_MAGIC_ADDR, magic);

    if (magic != EEPROM_MAGIC_VALUE) {
        // First run or corrupted: write defaults
        EEPROM.put(EEPROM_MAGIC_ADDR, (int16_t)EEPROM_MAGIC_VALUE);
        EEPROM.put(EEPROM_PAN_OFF_ADDR, (int16_t)panOffset);
        EEPROM.put(EEPROM_TILT_OFF_ADDR, (int16_t)tiltOffset);
        EEPROM.put(EEPROM_PAN_MIN_ADDR, (int16_t)panMin);
        EEPROM.put(EEPROM_PAN_MAX_ADDR, (int16_t)panMax);
        EEPROM.put(EEPROM_TILT_MIN_ADDR, (int16_t)tiltMin);
        EEPROM.put(EEPROM_TILT_MAX_ADDR, (int16_t)tiltMax);
        return;
    }

    int16_t val;
    EEPROM.get(EEPROM_PAN_OFF_ADDR, val);  panOffset = val;
    EEPROM.get(EEPROM_TILT_OFF_ADDR, val); tiltOffset = val;
    EEPROM.get(EEPROM_PAN_MIN_ADDR, val);  panMin = val;
    EEPROM.get(EEPROM_PAN_MAX_ADDR, val);  panMax = val;
    EEPROM.get(EEPROM_TILT_MIN_ADDR, val); tiltMin = val;
    EEPROM.get(EEPROM_TILT_MAX_ADDR, val); tiltMax = val;
}
