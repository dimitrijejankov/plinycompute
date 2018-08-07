// define the service that grabs the sets
(function (angular) {
    'use strict';

    angular.module('app.setsAll', [])
        .factory('setsAll', ['$http', function ($http) {
            return {
                get: function () {
                    return $http({
                        method: 'GET',
                        url: 'api/sets'
                    });
                }
            };
        }]);

}(angular));

// define the service that grabs a particular set
(function (angular) {
    'use strict';

    angular.module('app.set', [])
        .factory('set', ['$http', function ($http) {
            return {
                get: function (dbID, setID) {
                    return $http({
                        method: 'GET',
                        url: ('api/set/' + dbID + "." + setID)
                    });
                }
            };
        }]);

}(angular));