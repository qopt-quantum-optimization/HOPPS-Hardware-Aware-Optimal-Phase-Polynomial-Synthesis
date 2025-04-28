OPENQASM 2.0;
include "qelib1.inc";
qreg q[14];
creg c[14];

// moment 0
h q[6];
h q[8];
h q[5];
h q[9];

// moment 1
cz q[8], q[6];

// moment 2
h q[11];
cz q[5], q[6];
cx q[8], q[9];

// moment 3
cx q[9], q[8];

// moment 4
cx q[8], q[9];

// moment 5
cz q[8], q[6];
cz q[5], q[9];
cx q[10], q[11];

// moment 6
cz q[8], q[9];
cx q[4], q[5];
cx q[11], q[10];

// moment 7
x q[8];
cx q[5], q[4];
cx q[10], q[11];

// moment 8
cz q[9], q[10];
cx q[4], q[5];

// moment 9
cz q[4], q[10];
x q[9];

// moment 10
x q[6];
x q[10];
x q[4];

// measurement
measure q[6]->c[0];
measure q[10]->c[1];
measure q[9]->c[2];
measure q[4]->c[3];
measure q[8]->c[4];
