#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "catch2/catch.hpp"
#include "libdl/hello.h"
#include "libdl/mathutils.h"

TEST_CASE( "firsttests", "[includes]" ) {
    REQUIRE( mathutils("test") == 0 );
}

TEST_CASE( "first tests", "[includes]" ) {
    REQUIRE( 0 == 0 );
}

// int main(){
//     hello("tesst");
// }

// int main(){
//      int a = 0;
// }