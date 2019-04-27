#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "catch2/catch.hpp"
#include "libdl/hello.h"

TEST_CASE( "first  tests", "[includes]" ) {
    REQUIRE( hello("test") == 0 );
}
