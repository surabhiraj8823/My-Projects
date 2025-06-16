export interface Country {
    name: string;
    states: Array<{
      name: string;
      cities: string[];
    }>;
  }