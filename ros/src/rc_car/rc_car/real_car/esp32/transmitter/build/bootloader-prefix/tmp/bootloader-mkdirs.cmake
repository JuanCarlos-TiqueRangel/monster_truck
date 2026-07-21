# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/juan/lehigh_PhD/esp32/esp/esp-idf/components/bootloader/subproject"
  "/home/juan/lehigh_PhD/monster_truck/ros/src/rc_car/rc_car/real_car/esp32/transmitter/build/bootloader"
  "/home/juan/lehigh_PhD/monster_truck/ros/src/rc_car/rc_car/real_car/esp32/transmitter/build/bootloader-prefix"
  "/home/juan/lehigh_PhD/monster_truck/ros/src/rc_car/rc_car/real_car/esp32/transmitter/build/bootloader-prefix/tmp"
  "/home/juan/lehigh_PhD/monster_truck/ros/src/rc_car/rc_car/real_car/esp32/transmitter/build/bootloader-prefix/src/bootloader-stamp"
  "/home/juan/lehigh_PhD/monster_truck/ros/src/rc_car/rc_car/real_car/esp32/transmitter/build/bootloader-prefix/src"
  "/home/juan/lehigh_PhD/monster_truck/ros/src/rc_car/rc_car/real_car/esp32/transmitter/build/bootloader-prefix/src/bootloader-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/juan/lehigh_PhD/monster_truck/ros/src/rc_car/rc_car/real_car/esp32/transmitter/build/bootloader-prefix/src/bootloader-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/juan/lehigh_PhD/monster_truck/ros/src/rc_car/rc_car/real_car/esp32/transmitter/build/bootloader-prefix/src/bootloader-stamp${cfgdir}") # cfgdir has leading slash
endif()
