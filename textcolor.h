#pragma once

void textColor(int clr) {
  printf("\u001b[38;5;%dm",clr);
}

