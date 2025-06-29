OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
creg c[5];

// moment 0
cz q[2], q[4];

// moment 1
cx q[2], q[1];

// moment 2
cx q[3], q[2];

// moment 3
cx q[2], q[3];

// moment 4
cx q[3], q[2];

// moment 5
cx q[1], q[2];

// moment 6
cz q[2], q[4];

// moment 7
cx q[1], q[2];

// moment 8
cx q[1], q[2];

// moment 9
cx q[2], q[1];

// moment 10
cx q[1], q[2];

// moment 11
cx q[0], q[1];

// moment 12
cx q[3], q[2];
cx q[1], q[0];

// moment 13
cx q[3], q[2];
cx q[0], q[1];

// moment 14
cx q[2], q[1];

// moment 15
cz q[1], q[0];

// moment 16
cx q[2], q[1];

// moment 17
cx q[3], q[2];

// moment 18
cx q[1], q[2];

// moment 19
cx q[2], q[1];

// moment 20
cx q[1], q[2];

// moment 21
cz q[1], q[0];

// moment 22
cz q[3], q[2];
cx q[0], q[1];

// moment 23
cz q[2], q[4];
cx q[1], q[0];

// moment 24
cx q[0], q[1];

// moment 25
cx q[1], q[2];

// moment 26
cx q[2], q[1];

// moment 27
cx q[1], q[2];

// moment 28
cx q[0], q[1];

// moment 29
cx q[1], q[2];

// moment 30
cz q[2], q[4];

// moment 31
cx q[1], q[2];

// moment 32
cx q[0], q[1];

// measurement
measure q[3]->c[0];
measure q[0]->c[1];
measure q[1]->c[2];
measure q[2]->c[3];
measure q[4]->c[4];
