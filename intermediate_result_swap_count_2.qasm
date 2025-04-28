OPENQASM 2.0;
include "qelib1.inc";
qreg q[14];
creg c[14];

// moment 0
h q[3];
h q[13];
h q[11];
h q[2];
h q[12];

// moment 1
cz q[11], q[3];

// moment 2
cx q[12], q[11];

// moment 3
cx q[11], q[12];

// moment 4
cz q[2], q[3];
cx q[12], q[11];

// moment 5
cz q[11], q[3];
cz q[2], q[12];

// moment 6
cz q[11], q[12];
x q[3];
cx q[2], q[1];

// moment 7
x q[11];
cx q[1], q[2];

// moment 8
cz q[12], q[13];
cx q[2], q[1];

// moment 9
cz q[1], q[13];

// moment 10
x q[13];
x q[12];
x q[1];

// measurement
measure q[3]->c[0];
measure q[13]->c[1];
measure q[12]->c[2];
measure q[1]->c[3];
measure q[11]->c[4];
